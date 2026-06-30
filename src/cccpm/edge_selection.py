import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp, t, rankdata
from typing import Union

import torch

from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid
import statsmodels.stats.multitest as multitest
from warnings import filterwarnings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def one_sample_t_test(matrix, population_mean):
    # Calculate the mean and standard deviation along the rows
    sample_means = np.mean(matrix, axis=0)
    sample_stds = np.std(matrix, axis=0, ddof=1)
    n = matrix.shape[1]  # Number of samples in each row

    # Calculate the t-statistics
    filterwarnings('ignore', category=RuntimeWarning)
    t_stats = (sample_means - population_mean) / (sample_stds / np.sqrt(n))

    # Calculate the p-values using the t-distribution survival function
    p_values = 2 * t.sf(np.abs(t_stats), df=n - 1)

    return t_stats, p_values


def compute_t_and_p_values(correlations, df):
    # Calculate t-statistics
    t_stats = correlations * np.sqrt(df / (1 - correlations ** 2))
    # Calculate p-values
    p_values = 2 * t.sf(np.abs(t_stats), df=df)
    return t_stats, p_values


def compute_correlation_and_pvalues(x, Y, rank=False):
    n = len(x)

    if rank:
        x = rankdata(x)
        Y = rankdata(Y, axis=0)

    # Mean-centering
    x_centered = x - np.mean(x)
    Y_centered = Y - np.mean(Y, axis=0)

    # Correlation calculation
    corr_numerator = np.dot(Y_centered.T, x_centered)
    corr_denominator = (np.sqrt(np.sum(Y_centered ** 2, axis=0)) * np.sqrt(np.sum(x_centered ** 2)))

    correlations = corr_numerator / corr_denominator

    # Calculate t-statistics and p-values
    _, p_values = compute_t_and_p_values(correlations, n - 2)

    return correlations, p_values


def pearson_correlation_with_pvalues(x, Y):
    return compute_correlation_and_pvalues(x, Y, rank=False)


def spearman_correlation_with_pvalues(x, Y):
    return compute_correlation_and_pvalues(x, Y, rank=True)


def torch_rankdata(data, dim=-1):
    """
    Computes ranks of the data along a given dimension.
    Equivalent to scipy.stats.rankdata (method='ordinal') but fully vectorized on GPU.
    """
    # argsort twice gives the rank indices (0, 1, 2...)
    # We add 1.0 to match standard 1-based ranking
    return data.argsort(dim=dim).argsort(dim=dim).float() + 1.0


def get_residuals(data, confounds):
    """
    Regresses out 'confounds' from 'data' using OLS and returns the residuals.

    Args:
        data: (..., N_samples) or (N_samples, ...)
              The target data (can be X or Y). Can be numpy or torch.
        confounds: (N_samples, N_confounds). Can be numpy or torch.

    Returns:
        residuals: Same shape and type as data
    """
    # 1. Add Intercept column to confounds (standard OLS practice)
    # Shape: (N_samples, N_confounds + 1)
    if not hasattr(confounds, 'shape'):  # Safety check
        return data

    # Convert numpy to torch if needed, track for conversion back
    return_numpy = isinstance(data, np.ndarray)
    data = torch.as_tensor(data, dtype=torch.float64)
    confounds = torch.as_tensor(confounds, dtype=torch.float64)

    n_samples = confounds.shape[0]
    ones = torch.ones(n_samples, 1, dtype=confounds.dtype)
    Z = torch.cat((ones, confounds), dim=1)

    # 2. Compute the Projector (Hat Matrix component)
    # Beta = (Z^T Z)^-1 Z^T y
    # We precompute pinv(Z) for speed: (Z^T Z)^-1 Z^T
    # Z_pinv shape: (N_confounds+1, N_samples)
    Z_pinv = torch.linalg.pinv(Z)

    # 3. Apply to Data (Vectorized)
    # We need to handle data shapes carefully.
    # Data is usually [N_samples, Features] OR [N_perms, N_samples]

    def _maybe_to_numpy(result):
        return result.numpy() if return_numpy else result

    # CASE A: Data is [N_samples, Features] (Like X)
    if data.shape[0] == n_samples:
        # Beta: (Confounds, Features) = (Confounds, Samples) @ (Samples, Features)
        beta = torch.matmul(Z_pinv, data)
        # Preds: (Samples, Features) = (Samples, Confounds) @ (Confounds, Features)
        preds = torch.matmul(Z, beta)
        return _maybe_to_numpy(data - preds)

    # CASE B: Data is [Batch, N_samples] (Like Y_perms)
    elif data.shape[-1] == n_samples:
        # We assume data is [Batch, N_samples]. We need to transpose for matmul
        data_T = data.transpose(-1, -2)  # [N_samples, Batch]

        beta = torch.matmul(Z_pinv, data_T)  # [Confounds, Batch]
        preds = torch.matmul(Z, beta)  # [Samples, Batch]

        return _maybe_to_numpy((data_T - preds).transpose(-1, -2))  # Return to [Batch, Samples]

    else:
        raise ValueError(f"Data shape {data.shape} incompatible with confounds {confounds.shape}")


def correlations_and_pvalues(X, Y_perms,
                             correlation_type='pearson',
                             confounds=None):
    """
    Univariate edge selection as a vectorised OLS GLM, batched over permutations.

    For each edge (column of ``X``) and each (permuted) target (column of
    ``Y_perms``) this fits, in one batched linear-algebra pass on CPU/GPU, the
    linear model

        target ~ intercept [+ confounds] + edge

    and returns the edge effect and the p-value of its coefficient. By the
    Frisch–Waugh–Lovell theorem the coefficient only requires residualising the
    *edge* on the confounds — the target is **never** residualised to obtain the
    coefficient (its raw values drive it; this is what we want when the target is
    the thing we ultimately predict). This single path is mathematically
    identical to:

      * Pearson correlation                         (continuous target, no confounds)
      * point-biserial correlation                  (binary 0/1 target, no confounds)
      * the partial-correlation / coefficient F-test (confounds present)
      * Spearman, when ``X`` and ``Y`` are rank-transformed first.

    A binary (0/1) target needs no special handling: regressing it on an edge is
    the linear-probability model, whose coefficient test equals the
    point-biserial correlation. (Its homoskedastic p-values are the conventional
    point-biserial ones; OLS standard errors are not heteroskedasticity-robust,
    which is standard for this kind of screening filter.)

    The reported ``r`` is the **semi-partial** correlation (confounds removed
    from the connectome edge only, not the target). Its sign and the p-value are
    those of the regression coefficient, so the choice of semi-partial vs partial
    affects only the reported effect-size magnitude, never which edges are
    selected.

    Args:
        X: (N_samples, N_features) — continuous edge values (fixed across perms)
        Y_perms: (N_samples, N_perms) — target(s), one column per permutation
        correlation_type: 'pearson' (linear / point-biserial) or 'spearman' (ranks)
        confounds: optional (N_samples, N_confounds); when given, the edge effect
                   controls for these covariates.

    Returns:
        r_matrix: (N_features, N_perms) — semi-partial correlation (effect size)
        p_matrix: (N_features, N_perms) — p-value of the edge coefficient
    """
    X = torch.as_tensor(X)
    Y = torch.as_tensor(Y_perms, dtype=X.dtype, device=X.device)
    n_samples = X.size(0)

    # Spearman = Pearson on ranks. Rank every variable (incl. the confounds),
    # matching the conventional "rank, then partial" definition (e.g. pingouin).
    if correlation_type == 'spearman':
        X = torch_rankdata(X, dim=0)
        Y = torch_rankdata(Y, dim=0)
        if confounds is not None:
            confounds = torch_rankdata(
                torch.as_tensor(confounds, dtype=X.dtype, device=X.device), dim=0)

    # --- Residualise on the confounds (or just centre, when there are none) ---
    if confounds is not None:
        confounds = torch.as_tensor(confounds, dtype=X.dtype, device=X.device)
        k_confounds = confounds.size(1)
        # FWL: residualising only the EDGE is sufficient for the coefficient.
        # Y is residualised solely to obtain the full model's error variance.
        X_res = torch.as_tensor(get_residuals(X, confounds), dtype=X.dtype, device=X.device)
        Y_res = torch.as_tensor(get_residuals(Y, confounds), dtype=X.dtype, device=X.device)
    else:
        # Residualising on an intercept only is just mean-centring.
        k_confounds = 0
        X_res = X - X.mean(dim=0, keepdim=True)
        Y_res = Y - Y.mean(dim=0, keepdim=True)

    # Mean-centre so cross-products are (co)variances. X_res is already centred
    # (the intercept is part of the confound space), but be explicit.
    X_res = X_res - X_res.mean(dim=0, keepdim=True)
    Y_res = Y_res - Y_res.mean(dim=0, keepdim=True)
    Y_centered = Y - Y.mean(dim=0, keepdim=True)   # raw target, centred

    # Cross-products. Because X_res is orthogonal to the confound space
    # (including the intercept), X_res^T Y == X_res^T Y_res, so the raw centred
    # target gives exactly the regression coefficient (no Y residualisation).
    cross = torch.matmul(X_res.t(), Y_centered)        # (F, P)  = x_res^T y
    sxx = (X_res ** 2).sum(dim=0)                      # (F,)    ||x_res||^2
    sse_y = (Y_res ** 2).sum(dim=0)                    # (P,)    full-model error SS
    ssy = (Y_centered ** 2).sum(dim=0)                 # (P,)    total SS of target

    # Partial correlation == regression-coefficient test; it drives the p-value.
    partial_r = cross / (torch.sqrt(sxx.unsqueeze(1) * sse_y.unsqueeze(0)) + 1e-12)
    partial_r = torch.clamp(partial_r, -0.999999, 0.999999)

    # Semi-partial correlation (confounds removed from the edge only) — reported
    # effect size; same sign as the coefficient.
    semipartial_r = cross / (torch.sqrt(sxx.unsqueeze(1) * ssy.unsqueeze(0)) + 1e-12)
    semipartial_r = torch.clamp(semipartial_r, -0.999999, 0.999999)

    # p-value of the edge coefficient, df = N - k - 2.
    # NOTE: normal approximation to the t-tail, kept identical to the previous
    # implementation. See RELEASE_PLAN "Open decisions #6" — do not change the
    # tail approximation without sign-off (torch 2.x lacks an exact incomplete
    # beta / t-CDF; this is the GPU-friendly approximation).
    df = torch.tensor(n_samples - 2 - k_confounds, device=X.device, dtype=partial_r.dtype)
    t_stats = partial_r * torch.sqrt(df / (1 - partial_r ** 2))
    z = t_stats / torch.sqrt(df / (df + 1))
    val = -torch.abs(z) / 1.41421356
    p_matrix = 2 * (0.5 * (1 + torch.erf(val)))

    return semipartial_r, p_matrix



class BaseEdgeSelector(BaseEstimator):
    def select(self, r, p):
        pass


class PThreshold(BaseEdgeSelector):
    """
    Select edges whose correlation with the target is significant at a p-value
    threshold, optionally after multiple-comparison correction.

    Pass a list of thresholds (and/or corrections) to search over them with an
    inner cross-validation loop.
    """
    def __init__(self, threshold: Union[float, list] = 0.05, correction: Union[str, list] = None):
        """
        :param threshold: p-value threshold(s); edges with p below the threshold
                          are selected. A single value (e.g. ``0.05``) or a list
                          (e.g. ``[0.01, 0.05]``) to tune via inner CV.
        :param correction: multiple-comparison correction, or ``None`` for no
                            correction. Can be one of statsmodels' methods:
                            bonferroni : one-step correction
                            sidak : one-step correction
                            holm-sidak : step down method using Sidak adjustments
                            holm : step-down method using Bonferroni adjustments
                            simes-hochberg : step-up method (independent)
                            hommel : closed method based on Simes tests (non-negative)
                            fdr_bh : Benjamini/Hochberg (non-negative)
                            fdr_by : Benjamini/Yekutieli (negative)
                            bonferroni : one-step correction
                            sidak : one-step correction
                            holm-sidak : step down method using Sidak adjustments
                            holm : step-down method using Bonferroni adjustments
                            simes-hochberg : step-up method (independent)
                            hommel : closed method based on Simes tests (non-negative)
                            fdr_bh : Benjamini/Hochberg (non-negative)
                            fdr_by : Benjamini/Yekutieli (negative)
                            fdr_tsbh : two stage fdr correction (non-negative)
                            fdr_tsbky : two stage fdr correction (non-negative)
        """
        self._threshold = None
        self._correction = None
        self.threshold = threshold
        self.correction = correction

    @property
    def threshold(self):
        if isinstance(self._threshold, (int, float)):
            return [float(self._threshold)]
        return self._threshold or [0.05]

    @threshold.setter
    def threshold(self, value):
        if isinstance(value, (int, float)):
            self._threshold = float(value)
        elif isinstance(value, list):
            self._threshold = value
        else:
            raise ValueError("threshold must be float or list")

    @property
    def correction(self):
        if self._correction is None:
            return [None]
        if isinstance(self._correction, str):
            return [self._correction]
        return self._correction

    @correction.setter
    def correction(self, value):
        if value is None:
            self._correction = None
        elif isinstance(value, str):
            self._correction = value
        elif isinstance(value, list):
            self._correction = value
        else:
            raise ValueError("correction must be None, str, or list")

    def select(self, r, p):
        # Correction logic (requires p to be flat/numpy usually, ensure compatibility)
        if self._correction is not None:
            # Assuming p is passed as or converted to numpy for statsmodels
            from statsmodels.stats import multitest
            # You might need to flatten and reshape if p is multidimensional
            shape = p.shape
            _, p_flat, _, _ = multitest.multipletests(p.flatten(), alpha=0.05, method=self._correction)
            p = p_flat.reshape(shape)  # Reshape back or keep as tensor depending on input type

        # Calculate boolean masks
        pos_mask = (p < self.threshold[0]) & (r > 0)
        neg_mask = (p < self.threshold[0]) & (r < 0)

        # Stack into a single tensor: [Features, 2, ...]
        return torch.stack([torch.as_tensor(pos_mask, device=r.device),
                            torch.as_tensor(neg_mask, device=r.device)], dim=1)

class SelectPercentile(BaseEdgeSelector):
    def __init__(self, percentile: Union[float, list] = 0.05):
        self.percentile = percentile


class SelectKBest(BaseEdgeSelector):
    def __init__(self, k: Union[int, list] = None):
        self.k = k


class EdgeStatistic(BaseEstimator):
    def __init__(self, edge_statistic: str = 'spearman', t_test_filter: bool = False):
        self.edge_statistic = edge_statistic
        self.t_test_filter = t_test_filter

    def fit_transform(self,
                      X,
                      y,
                      covariates,
                      device):
        r_edges, p_edges = (torch.zeros((X.shape[1], y.shape[1]), device=device),
                            torch.ones((X.shape[1], y.shape[1]), device=device))
        #if self.t_test_filter:
        #    _, p_values = one_sample_t_test(X, 0)
        #    valid_edges = p_values < 0.05
        #else:
        #    valid_edges = np.bool(np.ones(X.shape[1]))

        # 1. Convert to GPU Tensors immediately
        X = torch.as_tensor(X, device=device, dtype=torch.float32)
        y = torch.as_tensor(y, device=device, dtype=torch.float32)
        if covariates is not None:
            covariates = torch.as_tensor(covariates, device=device, dtype=torch.float32)

        # 3. Variance Threshold (GPU Version)
        # Replaces sklearn.feature_selection.VarianceThreshold
        # Remove features with ~0 variance to avoid NaNs in correlation
        variances = torch.var(X, dim=0)
        valid_edges = variances > 1e-6

        if self.edge_statistic == 'pearson':
            r_edges_masked, p_edges_masked = correlations_and_pvalues(X=X[:, valid_edges], Y_perms=y,
                                                                      correlation_type='pearson')
        elif self.edge_statistic == 'spearman':
            r_edges_masked, p_edges_masked = correlations_and_pvalues(X=X[:, valid_edges], Y_perms=y,
                                                                      correlation_type='spearman')
        elif self.edge_statistic == 'pearson_partial':
            r_edges_masked, p_edges_masked = correlations_and_pvalues(X=X[:, valid_edges], Y_perms=y,
                                                                      confounds=covariates,
                                                                      correlation_type='pearson')
        elif self.edge_statistic == 'spearman_partial':
            r_edges_masked, p_edges_masked = correlations_and_pvalues(X=X[:, valid_edges], Y_perms=y,
                                                                      confounds=covariates,
                                                                      correlation_type='spearman')
        elif self.edge_statistic == 'point_biserial':
            # Point-biserial is Pearson against a binary 0/1 target; the unified
            # OLS path handles it with no special-casing.
            r_edges_masked, p_edges_masked = correlations_and_pvalues(X=X[:, valid_edges], Y_perms=y,
                                                                      correlation_type='pearson')
        elif self.edge_statistic == 'point_biserial_partial':
            r_edges_masked, p_edges_masked = correlations_and_pvalues(X=X[:, valid_edges], Y_perms=y,
                                                                      confounds=covariates,
                                                                      correlation_type='pearson')
        else:
            raise NotImplementedError("Unsupported edge selection method")
        r_edges[valid_edges] = r_edges_masked.to(r_edges.dtype)
        p_edges[valid_edges] = p_edges_masked.to(p_edges.dtype)
        return r_edges, p_edges


class UnivariateEdgeSelection(BaseEstimator):
    """
    Univariate edge selection for CPM.

    Correlates each edge with the target using the chosen statistic and selects
    edges with one or more selection strategies (e.g. a p-value threshold). When
    several configurations are supplied, they form a hyperparameter grid that the
    inner cross-validation loop searches over.

    Parameters
    ----------
    edge_statistic: str, default='spearman'
        Correlation statistic used to relate each edge to the target. One of
        ``'pearson'``, ``'spearman'``, ``'pearson_partial'``, ``'spearman_partial'``
        (continuous target), or ``'point_biserial'`` / ``'point_biserial_partial'``
        (binary target). The ``*_partial`` variants control for the covariates
        during selection.
    t_test_filter: bool, default=False
        Reserved for an optional pre-filtering step (currently inactive).
    edge_selection: list of selectors (e.g. PThreshold), default=None
        One or more selection strategies. Provide a list of multiple
        configurations to tune them via an inner CV.
    """
    def __init__(self,
                 edge_statistic: str = 'spearman',
                 t_test_filter: bool = False,
                 edge_selection: Union[list, None, PThreshold] = None):
        self.r_edges = None
        self.p_edges = None
        self.t_test_filter = t_test_filter
        self.edge_statistic = EdgeStatistic(edge_statistic=edge_statistic, t_test_filter=t_test_filter)
        self.edge_selection = edge_selection
        if isinstance(edge_selection, (list, tuple)):
            self.edge_selection = edge_selection
        else:
            self.edge_selection = [edge_selection]
        self.param_grid = self._generate_config_grid()

    def _generate_config_grid(self):
        grid_elements = []
        for selector in self.edge_selection:
            params = {}
            params['edge_selection'] = [selector]
            for key, value in selector.get_params().items():
                params['edge_selection__' + key] = value
            grid_elements.append(params)
        return ParameterGrid(grid_elements)

    def fit_transform(self, X, y=None, covariates=None, device=torch.device('cpu')):
        self.r_edges, self.p_edges = self.edge_statistic.fit_transform(X=X, y=y, covariates=covariates, device=device)
        return self

    def return_selected_edges(self):
        selected_edges = self.edge_selection.select(r=self.r_edges, p=self.p_edges)
        return selected_edges
