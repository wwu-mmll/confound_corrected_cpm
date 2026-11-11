"""The edge-level statistics: one vectorised OLS GLM underlies Pearson, Spearman, point-biserial and their partial variants.

p-values are exact (scipy, CPU) but too slow per edge x permutation, so the
pipeline thresholds the t statistic against `critical_t` instead (see
`PThreshold.select`), which selects exactly the edges an exact p-value would.
"""
import numpy as np
import torch
from scipy import special, stats


def torch_rankdata(data, dim=-1):
    """1-based ranks along `dim`; equivalent to scipy.stats.rankdata(method='ordinal') but GPU-vectorized (double argsort)."""
    return data.argsort(dim=dim).argsort(dim=dim).float() + 1.0


def get_residuals(data, confounds):
    """Regresses `confounds` out of `data` via OLS; returns residuals of the same shape/type (numpy or torch)."""
    if not hasattr(confounds, 'shape'):
        return data

    # Read dtype/device off the converted tensor, not the raw input: a numpy
    # dtype object is rejected by torch.as_tensor's dtype= argument.
    return_numpy = isinstance(data, np.ndarray)
    data = torch.as_tensor(data)
    dtype = data.dtype
    device = data.device

    confounds = torch.as_tensor(
        confounds,
        dtype=dtype,
        device=device
    )

    n_samples = confounds.shape[0]
    ones = torch.ones(n_samples, 1, dtype=confounds.dtype, device=confounds.device)
    Z = torch.cat((ones, confounds), dim=1)

    # Precompute pinv(Z) = (Z^T Z)^-1 Z^T once; beta = pinv(Z) @ y.
    Z_pinv = torch.linalg.pinv(Z)

    def _maybe_to_numpy(result):
        return result.cpu().numpy() if return_numpy else result

    if data.shape[0] == n_samples:
        # [N_samples, Features], e.g. X.
        beta = torch.matmul(Z_pinv, data)
        preds = torch.matmul(Z, beta)
        return _maybe_to_numpy(data - preds)
    elif data.shape[-1] == n_samples:
        # [Batch, N_samples], e.g. Y_perms; transpose for matmul.
        data_T = data.transpose(-1, -2)
        beta = torch.matmul(Z_pinv, data_T)
        preds = torch.matmul(Z, beta)
        return _maybe_to_numpy((data_T - preds).transpose(-1, -2))
    else:
        raise ValueError(f"Data shape {data.shape} incompatible with confounds {confounds.shape}")


def edge_statistics(X, Y_perms, correlation_type='pearson', confounds=None):
    """Fits ``target ~ intercept [+ confounds] + edge`` per edge x permutation column, batched on CPU/GPU.

    By Frisch-Waugh-Lovell, the coefficient only requires residualising the
    *edge* on the confounds -- the target is never residualised for it. This
    single path is mathematically identical to Pearson/point-biserial
    (no confounds), the partial-correlation coefficient test (confounds
    present), and Spearman (X, Y rank-transformed first); a binary 0/1 target
    needs no special case since its coefficient test equals point-biserial.

    Returns ``(r, t, df)`` where ``r`` is the **semi-partial** correlation
    (confounds removed from the edge only, not the target) of shape
    (N_features, N_perms); ``t`` same shape; ``df = N - 2 - N_confounds``.
    Semi-partial vs partial affects only the reported effect size, never
    which edges are selected (same sign/p-value as the coefficient).
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

    # FWL: residualising only the EDGE is sufficient for the coefficient; Y is
    # residualised solely to get the full model's error variance.
    if confounds is not None:
        confounds = torch.as_tensor(confounds, dtype=X.dtype, device=X.device)
        k_confounds = confounds.size(1)
        X_res = torch.as_tensor(get_residuals(X, confounds), dtype=X.dtype, device=X.device)
        Y_res = torch.as_tensor(get_residuals(Y, confounds), dtype=X.dtype, device=X.device)
    else:
        k_confounds = 0
        X_res = X - X.mean(dim=0, keepdim=True)
        Y_res = Y - Y.mean(dim=0, keepdim=True)

    X_res = X_res - X_res.mean(dim=0, keepdim=True)
    Y_res = Y_res - Y_res.mean(dim=0, keepdim=True)
    Y_centered = Y - Y.mean(dim=0, keepdim=True)

    # X_res is orthogonal to the confound space (incl. intercept), so
    # X_res^T Y == X_res^T Y_res -- the raw centred target gives exactly the
    # regression coefficient, with no need to residualise Y for it.
    cross = torch.matmul(X_res.t(), Y_centered)        # (F, P)  = x_res^T y
    sxx = (X_res ** 2).sum(dim=0)                      # (F,)    ||x_res||^2
    sse_y = (Y_res ** 2).sum(dim=0)                     # (P,)    full-model error SS
    ssy = (Y_centered ** 2).sum(dim=0)                  # (P,)    total SS of target

    # Partial correlation == regression-coefficient test; drives the t statistic.
    partial_r = cross / (torch.sqrt(sxx.unsqueeze(1) * sse_y.unsqueeze(0)) + 1e-12)
    partial_r = torch.clamp(partial_r, -0.999999, 0.999999)

    # Semi-partial (confounds removed from the edge only) -- the reported
    # effect size; same sign as the coefficient.
    semipartial_r = cross / (torch.sqrt(sxx.unsqueeze(1) * ssy.unsqueeze(0)) + 1e-12)
    semipartial_r = torch.clamp(semipartial_r, -0.999999, 0.999999)

    df = n_samples - 2 - k_confounds
    t_matrix = partial_r * torch.sqrt(df / (1 - partial_r ** 2))

    return semipartial_r, t_matrix, df


def t_pvalues(t, df):
    """Exact two-sided p-values (scipy, CPU); ~4s for 35,778 edges x 1000 permutations, why the pipeline uses `critical_t` instead."""
    t_np = t.detach().cpu().double().numpy()
    p = 2.0 * special.stdtr(df, -np.abs(t_np))
    return torch.as_tensor(p, dtype=t.dtype, device=t.device)


def critical_t(alpha, df):
    """The |t| above which a two-sided p-value falls below ``alpha`` (p is strictly decreasing in |t| for fixed df)."""
    return float(stats.t.isf(alpha / 2.0, df))


def correlations_and_pvalues(X, Y_perms, correlation_type='pearson', confounds=None):
    """`edge_statistics` with exact p-values in place of the t statistic; for direct use and tests."""
    r, t, df = edge_statistics(X, Y_perms, correlation_type=correlation_type,
                               confounds=confounds)
    return r, t_pvalues(t, df)
