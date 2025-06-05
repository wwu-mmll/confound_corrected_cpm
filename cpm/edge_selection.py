import pandas as pd
import torch
from scipy.stats import t as student_t
import numpy as np
from scipy.stats import ttest_1samp, t, rankdata

from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid
import statsmodels.stats.multitest as multitest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(arr, device):
    """pandas, numpy, Tensor → torch.Tensor on device."""
    if isinstance(arr, torch.Tensor):
        return arr.to(device)
    if isinstance(arr, (pd.Series, pd.DataFrame)):
        arr = arr.values
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr).float().to(device)
    if arr is None:
        return None
    raise ValueError(f"Cannot convert {type(arr)} to torch.Tensor")


def rankdata_1d(x: torch.Tensor) -> torch.Tensor:
    x = x.to(device)
    x_flat = x.flatten()
    n = x_flat.size(0)

    sorted_vals, sorted_idx = torch.sort(x_flat)
    unique_vals, inverse_idx = torch.unique(sorted_vals, return_inverse=True)
    rank_positions = torch.arange(
        1, n + 1, dtype=x.dtype, device=device
    )
    group_sum = torch.zeros_like(unique_vals, dtype=x.dtype, device=device)
    group_count = torch.zeros_like(unique_vals, dtype=x.dtype, device=device)
    group_sum.index_add_(0, inverse_idx, rank_positions)
    group_count.index_add_(0, inverse_idx, torch.ones_like(rank_positions))

    mean_ranks = group_sum / group_count
    ranks_sorted = mean_ranks[inverse_idx]

    ranks = torch.empty_like(ranks_sorted, device=device)
    ranks[sorted_idx] = ranks_sorted
    return ranks.view(x.shape)


def rankdata(a: torch.Tensor, axis: int = 0) -> torch.Tensor:
    a = a.to(device)
    if a.ndim == 1:
        return rankdata_1d(a)
    if axis != 0:
        raise NotImplementedError("Only axis=0 supported")
    out = torch.empty_like(a, dtype=torch.float32, device=device)
    for j in range(a.size(1)):
        out[:, j] = rankdata_1d(a[:, j])
    return out


def compute_t_and_p_values(corr: torch.Tensor, df: int):
    df_t = torch.tensor(df, dtype=corr.dtype, device=corr.device)
    t_stats = corr * torch.sqrt(df_t / (1 - corr ** 2))

    t_cpu = t_stats.detach().cpu().numpy()
    p_cpu = 2.0 * student_t.sf(np.abs(t_cpu), df)

    p_values = torch.from_numpy(p_cpu).to(corr.device).type_as(corr)
    return t_stats, p_values


def one_sample_t_test(matrix: torch.Tensor, population_mean: float):
    m = matrix.to(matrix.device)
    sample_means = m.mean(dim=0)
    sample_stds = m.std(dim=0, unbiased=True)
    n = m.size(0)

    t_stats = (sample_means - population_mean) / (
            sample_stds / torch.sqrt(torch.tensor(n, dtype=m.dtype, device=m.device))
    )
    df = n - 1

    t_cpu = t_stats.detach().cpu().numpy()
    p_cpu = 2.0 * student_t.sf(np.abs(t_cpu), df)

    p_values = torch.from_numpy(p_cpu).to(m.device).type_as(m)
    return t_stats, p_values


def compute_correlation_and_pvalues(
        x: torch.Tensor, Y: torch.Tensor, rank: bool = False
):
    x = x.to(device)
    Y = Y.to(device)
    if rank:
        x = rankdata(x)
        Y = rankdata(Y, axis=0)

    x_c = x - x.mean()
    Y_c = Y - Y.mean(dim=0)

    num = Y_c.T @ x_c
    den = torch.sqrt((Y_c ** 2).sum(dim=0) * (x_c ** 2).sum())
    corr = num / den

    _, pval = compute_t_and_p_values(corr, x.numel() - 2)
    return corr, pval


def get_residuals(X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    X = X.to(device)
    Z = Z.to(device)
    n, _ = Z.shape
    ones = torch.ones((n, 1), dtype=Z.dtype, device=device)
    Z_aug = torch.cat([Z, ones], dim=1)  # (n, cov+1)

    sol = torch.linalg.lstsq(Z_aug, X, rcond=None)
    B = sol.solution if hasattr(sol, "solution") else sol[0]
    X_hat = Z_aug @ B
    return X - X_hat


def semi_partial_correlation(x: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, rank: bool = False):
    x = x.to(device)
    Y = Y.to(device)
    Z = Z.to(device)

    if rank:
        x = rankdata(x)
        Y = rankdata(Y, axis=0)
        Z = rankdata(Z, axis=0)

    x_res = get_residuals(x.view(-1, 1), Z).ravel()
    Y_res = get_residuals(Y, Z)

    x_c = x_res - x_res.mean()
    Y_c = Y_res - Y_res.mean(dim=0)

    num = Y_c.T @ x_c
    den = torch.sqrt((Y_c ** 2).sum(dim=0) * (x_c ** 2).sum())
    corr = num / den

    n = x.numel()
    k = Z.size(1)
    _, pval = compute_t_and_p_values(corr, n - k - 2)
    return corr, pval


def pearson_correlation_with_pvalues(x, Y):
    return compute_correlation_and_pvalues(x, Y, rank=False)


def spearman_correlation_with_pvalues(x, Y):
    return compute_correlation_and_pvalues(x, Y, rank=True)


def semi_partial_correlation_pearson(x, Y, Z):
    return semi_partial_correlation(x, Y, Z, rank=False)


def semi_partial_correlation_spearman(x, Y, Z):
    return semi_partial_correlation(x, Y, Z, rank=True)


class BaseEdgeSelector(BaseEstimator):
    def select(self, r: torch.Tensor, p: torch.Tensor):
        raise NotImplementedError


class PThreshold(BaseEdgeSelector):
    def __init__(self, threshold=0.05, correction=None):
        self.threshold = threshold
        self.correction = correction

    def select(self, r: torch.Tensor, p: torch.Tensor):
        r = r.to(device)
        p = p.to(device)

        if self.correction is not None:
            p_corr = multitest.multipletests(
                p.cpu().numpy(), alpha=0.05, method=self.correction
            )[1]
            p = torch.from_numpy(p_corr).to(device)

        thr = torch.tensor(self.threshold, dtype=p.dtype, device=device)
        mask_pos = (p < thr) & (r > 0)
        mask_neg = (p < thr) & (r < 0)

        pos_idx = mask_pos.nonzero(as_tuple=True)[0]
        neg_idx = mask_neg.nonzero(as_tuple=True)[0]
        return {"positive": pos_idx, "negative": neg_idx}


class SelectPercentile(BaseEdgeSelector):
    def __init__(self, percentile=0.05):
        self.percentile = percentile

    def select(self, r, p):
        raise NotImplementedError


class SelectKBest(BaseEdgeSelector):
    def __init__(self, k=None):
        self.k = k

    def select(self, r, p):
        raise NotImplementedError


class EdgeStatistic(BaseEstimator):
    def __init__(
            self,
            edge_statistic: str = "spearman",
            t_test_filter: bool = False,
            device=None,
    ):
        self.edge_statistic = edge_statistic
        self.t_test_filter = t_test_filter
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit_transform(self, X: torch.Tensor, y: torch.Tensor, covariates: torch.Tensor):
        X = X.to(self.device)
        y = y.to(self.device)
        if covariates is not None:
            covariates = covariates.to(self.device)

        n_edges = X.size(1)
        r_edges = torch.zeros(n_edges, dtype=X.dtype, device=self.device)
        p_edges = torch.ones(n_edges, dtype=X.dtype, device=self.device)

        if self.t_test_filter:
            _, p0 = one_sample_t_test(X, 0.0)
            valid = p0 < 0.05
        else:
            valid = torch.ones(n_edges, dtype=torch.bool, device=self.device)

        X_valid = X[:, valid]
        if self.edge_statistic == "pearson":
            r_mask, p_mask = pearson_correlation_with_pvalues(y, X_valid)
        elif self.edge_statistic == "spearman":
            r_mask, p_mask = spearman_correlation_with_pvalues(y, X_valid)
        elif self.edge_statistic == "pearson_partial":
            r_mask, p_mask = semi_partial_correlation_pearson(y, X_valid, covariates)
        elif self.edge_statistic == "spearman_partial":
            r_mask, p_mask = semi_partial_correlation_spearman(y, X_valid, covariates)
        else:
            raise NotImplementedError(f"Unsupported method {self.edge_statistic}")

        r_edges[valid] = r_mask
        p_edges[valid] = p_mask
        return r_edges, p_edges


class UnivariateEdgeSelection(BaseEstimator):
    def __init__(self,
                 edge_statistic: str = 'spearman',
                 t_test_filter: bool = False,
                 edge_selection: BaseEdgeSelector = None,
                 device=None):

        self.edge_statistic = edge_statistic
        self.t_test_filter = t_test_filter
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.edge_selection = edge_selection or PThreshold()
        self.stat = EdgeStatistic(
            edge_statistic=edge_statistic,
            t_test_filter=t_test_filter,
            device=self.device
        )

        self.param_grid = self._generate_config_grid()

    def _generate_config_grid(self):
        sel = self.edge_selection
        params = {"edge_selection": [sel]}
        for k, v in sel.get_params().items():
            params[f"edge_selection__{k}"] = v if isinstance(v, list) else [v]
        return list(ParameterGrid([params]))

    def fit_transform(self, X, y=None, covariates=None):
        X_t = _to_tensor(X, self.device)
        y_t = _to_tensor(y, self.device)
        cov_t = _to_tensor(covariates, self.device)

        self.r_edges, self.p_edges = self.stat.fit_transform(X=X_t, y=y_t, covariates=cov_t)
        return self

    def return_selected_edges(self):
        edges = self.edge_selection.select(r=self.r_edges, p=self.p_edges)
        for net, idxs in edges.items():
            # Torch-Tensor → CPU → NumPy → int
            if isinstance(idxs, torch.Tensor):
                idxs = idxs.cpu().numpy().astype(int)
            if isinstance(idxs, np.ndarray):
                idxs = idxs.astype(int)
            edges[net] = idxs.tolist() if isinstance(idxs, np.ndarray) else list(idxs)
        return edges
