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


def get_residuals(X, Z):
    # Add a column of ones to Z for the intercept
    Z = np.hstack([Z, np.ones((Z.shape[0], 1))])

    # Compute the coefficients using the normal equation
    B = np.linalg.lstsq(Z, X, rcond=None)[0]

    # Predict X from Z
    X_hat = Z.dot(B)

    # Compute residuals
    residuals = X - X_hat

    return residuals


def semi_partial_correlation(x, Y, Z, rank=False):
    # ToDo: THIS IS A PARTIAL CORRELATION, NOT SEMI-PARTIAL
    if rank:
        x = rankdata(x)
        Y = rankdata(Y, axis=0)
        Z = rankdata(Z, axis=0)

        #Y = np.apply_along_axis(rankdata, 0, Y)
        #Z = np.apply_along_axis(rankdata, 0, Z)

    # Calculate residuals for x and each column in Y
    x_residuals = get_residuals(x.reshape(-1, 1), Z).ravel()
    Y_residuals = get_residuals(Y, Z)

    # Mean-centering the residuals
    x_centered = x_residuals - np.mean(x_residuals)
    Y_centered = Y_residuals - np.mean(Y_residuals, axis=0)

    # Correlation calculation
    corr_numerator = np.dot(Y_centered.T, x_centered)
    corr_denominator = (np.sqrt(np.sum(Y_centered ** 2, axis=0)) * np.sqrt(np.sum(x_centered ** 2)))
    partial_corr = corr_numerator / corr_denominator

    # Calculate t-statistics and p-values
    n = len(x)
    k = Z.shape[1]
    _, p_values = compute_t_and_p_values(partial_corr, n - k - 2)

    return partial_corr, p_values


def pearson_correlation_with_pvalues(x, Y):
    return compute_correlation_and_pvalues(x, Y, rank=False)


def spearman_correlation_with_pvalues(x, Y):
    return compute_correlation_and_pvalues(x, Y, rank=True)


def semi_partial_correlation_pearson(x, Y, Z):
    return semi_partial_correlation(x, Y, Z, rank=False)


def semi_partial_correlation_spearman(x, Y, Z):
    return semi_partial_correlation(x, Y, Z, rank=True)


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
              The target data (can be X or Y).
        confounds: (N_samples, N_confounds)

    Returns:
        residuals: Same shape as data
    """
    # 1. Add Intercept column to confounds (standard OLS practice)
    # Shape: (N_samples, N_confounds + 1)
    if not hasattr(confounds, 'shape'):  # Safety check
        return data

    n_samples = confounds.shape[0]
    ones = torch.ones(n_samples, 1, device=confounds.device, dtype=confounds.dtype)
    Z = torch.cat((ones, confounds), dim=1)

    # 2. Compute the Projector (Hat Matrix component)
    # Beta = (Z^T Z)^-1 Z^T y
    # We precompute pinv(Z) for speed: (Z^T Z)^-1 Z^T
    # Z_pinv shape: (N_confounds+1, N_samples)
    Z_pinv = torch.linalg.pinv(Z)

    # 3. Apply to Data (Vectorized)
    # We need to handle data shapes carefully.
    # Data is usually [N_samples, Features] OR [N_perms, N_samples]

    # CASE A: Data is [N_samples, Features] (Like X)
    if data.shape[0] == n_samples:
        # Beta: (Confounds, Features) = (Confounds, Samples) @ (Samples, Features)
        beta = torch.matmul(Z_pinv, data)
        # Preds: (Samples, Features) = (Samples, Confounds) @ (Confounds, Features)
        preds = torch.matmul(Z, beta)
        return data - preds

    # CASE B: Data is [Batch, N_samples] (Like Y_perms)
    elif data.shape[-1] == n_samples:
        # We assume data is [Batch, N_samples]. We need to transpose for matmul
        data_T = data.transpose(-1, -2)  # [N_samples, Batch]

        beta = torch.matmul(Z_pinv, data_T)  # [Confounds, Batch]
        preds = torch.matmul(Z, beta)  # [Samples, Batch]

        return (data_T - preds).transpose(-1, -2)  # Return to [Batch, Samples]

    else:
        raise ValueError(f"Data shape {data.shape} incompatible with confounds {confounds.shape}")


def point_biserial_correlation(X, Y_perms, confounds=None):
    """
    Computes point-biserial correlation for binary target variables.

    Point-biserial correlation measures the relationship between a continuous
    variable (features in X) and a binary variable (Y). It is mathematically
    equivalent to Pearson correlation when one variable is binary.

    Args:
        X: (N_samples, N_features) - Continuous features
        Y_perms: (N_samples, N_perms) - Binary target (0 or 1)
        confounds: Optional (N_samples, N_confounds) tensor for partial correlation

    Returns:
        r_matrix: (N_features, N_perms) - Point-biserial correlations
        p_matrix: (N_features, N_perms) - P-values
    """
    n_samples = X.size(0)

    # Handle partial correlation (residualize first)
    if confounds is not None:
        confounds = torch.from_numpy(confounds) if isinstance(confounds, np.ndarray) else confounds
        X = get_residuals(X, confounds)
        Y_perms = get_residuals(Y_perms, confounds)
        k_confounds = confounds.size(1)
    else:
        k_confounds = 0

    # Y_perms shape: (N_samples, N_perms)
    # Create masks for binary groups (0 and 1)
    mask_1 = (Y_perms == 1).unsqueeze(1)  # (N_samples, 1, N_perms)
    mask_0 = (Y_perms == 0).unsqueeze(1)  # (N_samples, 1, N_perms)

    # Expand X for broadcasting: (N_samples, N_features) -> (N_samples, N_features, 1)
    X_expanded = X.unsqueeze(2)

    # Calculate group statistics
    # Sum and count for each group
    n1 = mask_1.sum(dim=0)  # (1, N_perms)
    n0 = mask_0.sum(dim=0)  # (1, N_perms)

    # Mean for each group and feature
    sum_1 = (X_expanded * mask_1).sum(dim=0)  # (N_features, N_perms)
    sum_0 = (X_expanded * mask_0).sum(dim=0)  # (N_features, N_perms)

    M1 = sum_1 / (n1 + 1e-8)  # (N_features, N_perms)
    M0 = sum_0 / (n0 + 1e-8)  # (N_features, N_perms)

    # Standard deviation (pooled)
    # Var = E[X^2] - E[X]^2
    X_squared = X_expanded ** 2
    sum_sq_1 = (X_squared * mask_1).sum(dim=0)
    sum_sq_0 = (X_squared * mask_0).sum(dim=0)

    # Pooled variance
    ss_1 = sum_sq_1 - (sum_1 ** 2) / (n1 + 1e-8)
    ss_0 = sum_sq_0 - (sum_0 ** 2) / (n0 + 1e-8)
    pooled_var = (ss_1 + ss_0) / (n1 + n0 - 2 + 1e-8)
    s = torch.sqrt(pooled_var)

    # Point-biserial correlation formula:
    # r_pb = (M1 - M0) / s * sqrt(n0 * n1 / (n0 + n1)^2)
    r_matrix = (M1 - M0) / (s + 1e-8) * torch.sqrt(n0 * n1 / ((n0 + n1) ** 2 + 1e-8))

    # Clamp to valid correlation range
    r_matrix = torch.clamp(r_matrix, -0.999999, 0.999999)

    # Calculate p-values using t-distribution
    df = torch.tensor(n_samples - 2 - k_confounds, device=X.device, dtype=X.dtype)
    t_stats = r_matrix * torch.sqrt(df / (1 - r_matrix ** 2))

    # Approximate p-values using normal distribution (fast approximation)
    z = t_stats / torch.sqrt(df / (df + 1))
    val = -torch.abs(z) / 1.41421356
    p_matrix = 2 * (0.5 * (1 + torch.erf(val)))

    return r_matrix, p_matrix  # Shape: (N_features, N_perms)


def correlations_and_pvalues(X, Y_perms,
                             correlation_type='pearson',
                             confounds=None):
    """
    Computes correlations (Pearson/Spearman/Partial/Point-Biserial) and p-values on GPU.

    Args:
        X: (N_samples, N_features)
        Y_perms: (N_perms, N_samples)
        correlation_type: 'pearson', 'spearman', or 'point_biserial'
        confounds: Optional (N_samples, N_confounds) tensor.
                   If provided, partial correlation is computed.
    """
    # Check if Y is binary and auto-select point-biserial if appropriate
    Y_unique = torch.unique(Y_perms)
    is_binary = len(Y_unique) == 2 and set(Y_unique.cpu().numpy()).issubset({0, 1, -1})

    if is_binary and correlation_type in ['pearson', 'spearman']:
        # Binary target detected - use point-biserial (equivalent to Pearson for binary)
        # Convert -1/1 to 0/1 if needed
        if -1 in Y_unique:
            Y_perms = (Y_perms + 1) / 2
        return point_biserial_correlation(X, Y_perms, confounds)

    n_samples = X.size(0)

    # --- A. HANDLE PARTIAL CORRELATION (RESIDUALIZE) ---
    if confounds is not None:
        confounds = torch.from_numpy(confounds)
        # 1. Clean X (Fixed)
        X = get_residuals(X, confounds)
        # 2. Clean Y (Batch)
        Y_perms = get_residuals(Y_perms, confounds)

        # Note: When doing partial correlation, Degrees of Freedom decrease
        # df = N - 2 - k (where k is number of confounds)
        k_confounds = confounds.size(1)
    else:
        k_confounds = 0

    # --- B. HANDLE SPEARMAN (RANKING) ---
    if correlation_type == 'spearman':
        # Rank X (column-wise)
        X = torch_rankdata(X, dim=0)
        # Rank Y (row-wise for each perm)
        Y_perms = torch_rankdata(Y_perms, dim=0)

    # --- C. STANDARD PEARSON LOGIC (Now applied to Ranks or Residuals) ---

    # 1. Standardize Inputs (Z-score)
    X_mean = X.mean(dim=0, keepdim=True)
    X_std = X.std(dim=0, keepdim=True)
    X_norm = (X - X_mean) / (X_std + 1e-8)

    Y_mean = Y_perms.mean(dim=0, keepdim=True)
    Y_std = Y_perms.std(dim=0, keepdim=True)
    Y_norm = (Y_perms - Y_mean) / (Y_std + 1e-8)

    # 2. Correlation (MatMul)
    r_matrix = torch.matmul(Y_norm.t(), X_norm) / (n_samples - 1)

    # 3. P-Values
    r_matrix = torch.clamp(r_matrix, -0.999999, 0.999999)

    # Update DF based on confounds
    df = torch.tensor(n_samples - 2 - k_confounds, device=X.device, dtype=X.dtype)

    t_stats = r_matrix * torch.sqrt(df / (1 - r_matrix ** 2))
    z = t_stats / torch.sqrt(df / (df + 1))
    val = -torch.abs(z) / 1.41421356
    p_matrix = 2 * (0.5 * (1 + torch.erf(val)))

    return r_matrix.t(), p_matrix.t()



class BaseEdgeSelector(BaseEstimator):
    def select(self, r, p):
        pass


class PThreshold(BaseEdgeSelector):
    def __init__(self, threshold: Union[float, list] = 0.05, correction: Union[str, list] = None):
        """

        :param threshold:
        :param correction: can be one of statsmodels methods
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
            r_edges_masked, p_edges_masked = point_biserial_correlation(X=X[:, valid_edges], Y_perms=y,
                                                                        confounds=None)
        elif self.edge_statistic == 'point_biserial_partial':
            r_edges_masked, p_edges_masked = point_biserial_correlation(X=X[:, valid_edges], Y_perms=y,
                                                                        confounds=covariates)
        else:
            raise NotImplementedError("Unsupported edge selection method")
        r_edges[valid_edges] = r_edges_masked
        p_edges[valid_edges] = p_edges_masked
        return r_edges, p_edges


class UnivariateEdgeSelection(BaseEstimator):
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
