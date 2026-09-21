"""
The edge-level statistics.

One vectorised OLS GLM underlies every edge statistic CCCPM offers -- Pearson,
Spearman, point-biserial and their partial variants are all the same fit with
different preprocessing of the inputs (see `correlations_and_pvalues`). This
module is that fit, plus the two transforms it composes with: rank conversion
and confound residualisation, and the Bonferroni correction applied to its
p-values.

Split out of `edge_selection.py` so that the maths sits apart from the policy
that consumes it -- which edges to keep, and how the selectors are configured.
"""
import numpy as np
import torch


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

    # Convert numpy to torch if needed, track for conversion back. Convert
    # `data` first, then read dtype/device off the resulting tensor (not the
    # raw input): a numpy array's `.dtype` is a numpy dtype object, which
    # torch.as_tensor's `dtype=` argument rejects.
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
    # Create the intercept column on the same device as the inputs so this works
    # when X/confounds live on the GPU (otherwise torch.cat mixes cpu + cuda).
    ones = torch.ones(n_samples, 1, dtype=confounds.dtype, device=confounds.device)
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
        # .cpu() is a no-op on CPU tensors but required before .numpy() on GPU.
        return result.cpu().numpy() if return_numpy else result

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


def torch_bonferroni(p, alpha=0.05):
    """
    Bonferroni multiple-comparisons correction, computed entirely in torch.

    Equivalent to
    ``statsmodels.stats.multitest.multipletests(p.flatten(), alpha=alpha, method='bonferroni')``,
    but stays on-device (no CPU/GPU sync): p_corrected = min(p * n, 1), where n
    is the total number of tests (i.e. ``p.numel()``, matching statsmodels'
    behaviour of correcting over the flattened array).

    Args:
        p: p-values tensor, any shape.
        alpha: significance level used for the reject mask.

    Returns:
        reject: bool tensor, same shape as p — True where p_corrected < alpha.
        p_corrected: tensor, same shape as p, clamped to at most 1.
    """
    n = p.numel()
    p_corrected = torch.clamp(p * n, max=1.0)
    reject = p_corrected < alpha
    return reject, p_corrected


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
