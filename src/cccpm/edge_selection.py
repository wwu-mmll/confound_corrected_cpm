import numpy as np
from typing import Union

import networkx as nx
import torch

from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid


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


def _masked_rankdata(data, valid_mask):
    """
    Batched-over-folds rank transform (see utils.build_fold_batch): ranks
    each fold's REAL (valid) rows along the samples axis (dim 1), ignoring
    padded rows.

    Padded rows are set to +inf before ranking, so they always sort after
    every real value and never disturb the ranks of real values (a real
    element's rank among reals is unaffected by appending sentinels that
    always sort last), then masked back to exactly 0 afterward.

    Args:
        data: [B, N_max, F] -- padded, samples axis = dim 1.
        valid_mask: [B, N_max] bool.

    Returns:
        ranks: same shape as `data`; padded rows 0, real rows ranked 1..n_valid[b].
    """
    mask = valid_mask.unsqueeze(-1)  # [B, N_max, 1]
    sentinel = torch.full_like(data, float('inf'))
    data_for_rank = torch.where(mask, data, sentinel)
    ranks = data_for_rank.argsort(dim=1).argsort(dim=1).to(data.dtype) + 1.0
    return ranks * mask.to(data.dtype)


def get_residuals_batched(data, confounds, valid_mask):
    """
    Batched-over-folds version of `get_residuals` for zero-padded, ragged
    fold batches (see utils.build_fold_batch). Mathematically identical to
    calling `get_residuals` once per fold on that fold's real (unpadded)
    rows: padding is purely a memory-layout device, made inert by re-masking
    after every step that could disturb it.

    Args:
        data: [B, N_max, Features_or_Perms] -- padded, samples axis = dim 1.
        confounds: [B, N_max, N_confounds] -- padded, samples axis = dim 1.
        valid_mask: [B, N_max] bool -- True where the row is a real (non-pad) sample.

    Returns:
        residuals: same shape as `data`, padded rows exactly 0.
    """
    dtype = data.dtype
    mask = valid_mask.to(dtype).unsqueeze(-1)  # [B, N_max, 1]

    # The intercept column must ALSO be 0 on padded rows -- otherwise a
    # padded row of Z would be [1, 0, ..., 0] (not all-zero), which would
    # corrupt Z^T Z. Using the mask itself as the intercept column achieves
    # this "masked ones" property directly.
    Z = torch.cat([mask, confounds], dim=-1)  # [B, N_max, C+1], padded rows all-zero

    # Since padded rows of Z are exactly 0, Z^T Z naturally excludes their
    # contribution -- the batched pinv equals the pinv computed on each
    # fold's real rows only.
    Z_pinv = torch.linalg.pinv(Z)              # [B, C+1, N_max]

    beta = torch.matmul(Z_pinv, data)          # [B, C+1, Features_or_Perms]
    preds = torch.matmul(Z, beta)              # [B, N_max, Features_or_Perms]
    return (data - preds) * mask


def correlations_and_pvalues_batched(X, Y_perms, valid_mask,
                                      correlation_type='pearson', confounds=None):
    """
    Batched-over-folds version of `correlations_and_pvalues` for zero-padded,
    ragged fold batches (see utils.build_fold_batch). Mathematically
    identical to calling `correlations_and_pvalues` once per fold on that
    fold's real (unpadded) rows -- see `get_residuals_batched`/`_masked_rankdata`
    for how padding is kept inert throughout.

    Args:
        X: [B, N_max, N_features] -- padded, samples axis = dim 1.
        Y_perms: [B, N_max, N_perms] -- padded, samples axis = dim 1.
        valid_mask: [B, N_max] bool.
        correlation_type: 'pearson' or 'spearman'.
        confounds: optional [B, N_max, N_confounds] -- padded.

    Returns:
        r_matrix, p_matrix: [N_features, B_folds, N_perms].
    """
    X = torch.as_tensor(X)
    Y = torch.as_tensor(Y_perms, dtype=X.dtype, device=X.device)
    valid_mask = valid_mask.to(X.device)
    mask = valid_mask.to(X.dtype).unsqueeze(-1)   # [B, N_max, 1]
    n_valid = valid_mask.sum(dim=1).to(X.dtype)   # [B]

    if correlation_type == 'spearman':
        X = _masked_rankdata(X, valid_mask)
        Y = _masked_rankdata(Y, valid_mask)
        if confounds is not None:
            confounds = _masked_rankdata(
                torch.as_tensor(confounds, dtype=X.dtype, device=X.device), valid_mask)

    if confounds is not None:
        confounds = torch.as_tensor(confounds, dtype=X.dtype, device=X.device)
        k_confounds = confounds.size(-1)
        X_res = get_residuals_batched(X, confounds, valid_mask)
        Y_res = get_residuals_batched(Y, confounds, valid_mask)
    else:
        k_confounds = 0
        X_mean = (X * mask).sum(dim=1, keepdim=True) / n_valid.view(-1, 1, 1)
        Y_mean = (Y * mask).sum(dim=1, keepdim=True) / n_valid.view(-1, 1, 1)
        X_res = (X - X_mean) * mask
        Y_res = (Y - Y_mean) * mask

    # Redundant re-centring (X_res/Y_res are already centred), kept for
    # numerical parity with the unbatched implementation.
    X_mean2 = (X_res * mask).sum(dim=1, keepdim=True) / n_valid.view(-1, 1, 1)
    Y_mean2 = (Y_res * mask).sum(dim=1, keepdim=True) / n_valid.view(-1, 1, 1)
    X_res = (X_res - X_mean2) * mask
    Y_res = (Y_res - Y_mean2) * mask

    Y_mean_raw = (Y * mask).sum(dim=1, keepdim=True) / n_valid.view(-1, 1, 1)
    Y_centered = (Y - Y_mean_raw) * mask

    cross = torch.matmul(X_res.transpose(-1, -2), Y_centered)  # [B, F, P]
    sxx = (X_res ** 2).sum(dim=1)     # [B, F]
    sse_y = (Y_res ** 2).sum(dim=1)   # [B, P]
    ssy = (Y_centered ** 2).sum(dim=1)  # [B, P]

    partial_r = cross / (torch.sqrt(sxx.unsqueeze(-1) * sse_y.unsqueeze(1)) + 1e-12)
    partial_r = torch.clamp(partial_r, -0.999999, 0.999999)

    semipartial_r = cross / (torch.sqrt(sxx.unsqueeze(-1) * ssy.unsqueeze(1)) + 1e-12)
    semipartial_r = torch.clamp(semipartial_r, -0.999999, 0.999999)

    df = (n_valid - 2 - k_confounds).view(-1, 1, 1)  # [B, 1, 1], broadcasts vs [B, F, P]
    t_stats = partial_r * torch.sqrt(df / (1 - partial_r ** 2))
    z = t_stats / torch.sqrt(df / (df + 1))
    val = -torch.abs(z) / 1.41421356
    p_matrix = 2 * (0.5 * (1 + torch.erf(val)))

    # [B, F, P] -> [F, B, P] to match the (Features, Folds, Perms) convention.
    r_matrix = semipartial_r.permute(1, 0, 2)
    p_matrix = p_matrix.permute(1, 0, 2)
    return r_matrix, p_matrix


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



def resolve_presence_threshold(presence_filter):
    """
    Resolve the ``presence_filter`` argument to a nonzero-fraction threshold.

    Returns ``None`` when the filter is off, otherwise the fraction of subjects
    that must have a nonzero value for an edge to be kept (``True`` -> ``0.5``).
    """
    if presence_filter is False or presence_filter is None:
        return None
    if presence_filter is True:
        return 0.5
    threshold = float(presence_filter)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"presence_filter must be a bool or a fraction in [0, 1], "
            f"got {presence_filter!r}."
        )
    return threshold


def resolve_min_component_size(connected_components):
    """
    Resolve the ``connected_components`` argument to a minimum component size
    (in edges), or ``None`` when the filter is off (``True`` -> ``2``, i.e. drop
    lone single edges).
    """
    if connected_components is False or connected_components is None:
        return None
    if connected_components is True:
        return 2
    size = int(connected_components)
    if size < 1:
        raise ValueError(
            f"connected_components must be a bool or an int >= 1, "
            f"got {connected_components!r}."
        )
    return size


def filter_connected_components(mask, min_edges):
    """
    Keep only selected edges that belong to a connected component with at least
    ``min_edges`` edges, per network layer and run; drop the rest.

    Edges are nodes-in-common connections of a graph built per (network, run)
    from the selected edges. Isolated single edges form a one-edge component and
    are removed when ``min_edges >= 2``. ``mask`` is the ``[Features, 2, Runs]``
    selection tensor (dim 1 = positive/negative); the returned tensor has the
    same shape/dtype with dropped edges set to 0.
    """
    from cccpm.utils import infer_n_nodes

    n_features = mask.shape[0]
    n_nodes = infer_n_nodes(n_features)
    if n_nodes is None:
        return mask

    rows, cols = np.triu_indices(n_nodes, k=1)
    out = mask.clone()
    selected = mask.detach().cpu().numpy() > 0
    for layer in range(selected.shape[1]):
        for run in range(selected.shape[2]):
            edge_idx = np.nonzero(selected[:, layer, run])[0]
            if edge_idx.size == 0:
                continue
            graph = nx.Graph()
            graph.add_edges_from(zip(rows[edge_idx].tolist(), cols[edge_idx].tolist()))
            drop = set()
            for component in nx.connected_components(graph):
                sub = graph.subgraph(component)
                if sub.number_of_edges() < min_edges:
                    drop.update(tuple(sorted(e)) for e in sub.edges())
            if drop:
                for e in edge_idx:
                    if (rows[e], cols[e]) in drop:
                        out[e, layer, run] = 0
    return out


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

    def _apply_correction(self, p):
        """Multiple-comparison-correct `p`, returning a tensor of the same shape."""
        if self._correction is None:
            return p
        if self._correction == 'bonferroni':
            # Stays on-device (no CPU/GPU sync), unlike the statsmodels path below.
            _, p_corrected = torch_bonferroni(p)
            return p_corrected
        # Other statsmodels corrections still require a CPU round-trip.
        from statsmodels.stats import multitest
        shape = p.shape
        p_np = p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else p
        _, p_flat, _, _ = multitest.multipletests(p_np.flatten(), alpha=0.05, method=self._correction)
        p_flat = p_flat.reshape(shape)
        return torch.as_tensor(p_flat, device=p.device, dtype=p.dtype) if isinstance(p, torch.Tensor) else p_flat

    def select(self, r, p):
        p_corrected = self._apply_correction(p)

        # Calculate boolean masks
        pos_mask = (p_corrected < self.threshold[0]) & (r > 0)
        neg_mask = (p_corrected < self.threshold[0]) & (r < 0)

        # Stack into a single tensor: [Features, 2, ...]
        return torch.stack([torch.as_tensor(pos_mask, device=r.device),
                            torch.as_tensor(neg_mask, device=r.device)], dim=1)

    def select_batch(self, r, p, thresholds=None):
        """
        Batched version of `select()`: apply this selector's correction to
        `(r, p)` once, then compare against multiple threshold values at
        once, adding a "params" batch dimension to the returned edge mask
        instead of requiring one `select()` call per threshold. All
        thresholds must share this selector's single `correction` method
        (different correction methods need separate calls, since they use
        different formulas).

        Args:
            r, p: [N_features, *rest] (rest = any already-present batch dims,
                  e.g. folds/perms).
            thresholds: sequence of float p-value thresholds. Defaults to
                        `self.threshold` (already a list).

        Returns:
            Boolean tensor [N_features, 2, N_params, *rest], N_params = len(thresholds).
        """
        if thresholds is None:
            thresholds = self.threshold

        p_corrected = self._apply_correction(p)
        rest_shape = p_corrected.shape[1:]
        thresholds_t = torch.as_tensor(list(thresholds), device=r.device, dtype=p_corrected.dtype)
        thresh_view = thresholds_t.view(1, -1, *([1] * len(rest_shape)))

        p_exp = p_corrected.unsqueeze(1)  # [N_features, 1, *rest]
        r_exp = r.unsqueeze(1)            # [N_features, 1, *rest]
        pos_mask = (p_exp < thresh_view) & (r_exp > 0)
        neg_mask = (p_exp < thresh_view) & (r_exp < 0)

        # Stack into a single tensor: [Features, 2, N_params, *rest]
        return torch.stack([pos_mask, neg_mask], dim=1)


# edge_statistic name -> (correlation_type, use_confounds). Shared by
# EdgeStatistic.fit_transform_batched; fit_transform's own if/elif chain is
# left untouched to avoid any risk to its already-verified behaviour.
_EDGE_STATISTIC_DISPATCH = {
    'pearson': ('pearson', False),
    'spearman': ('spearman', False),
    'pearson_partial': ('pearson', True),
    'spearman_partial': ('spearman', True),
    'point_biserial': ('pearson', False),
    'point_biserial_partial': ('pearson', True),
}


class EdgeStatistic(BaseEstimator):
    def __init__(self, edge_statistic: str = 'spearman',
                 presence_filter: Union[bool, float] = False):
        self.edge_statistic = edge_statistic
        self.presence_filter = presence_filter

    def fit_transform(self,
                      X,
                      y,
                      covariates,
                      device):
        r_edges, p_edges = (torch.zeros((X.shape[1], y.shape[1]), device=device),
                            torch.ones((X.shape[1], y.shape[1]), device=device))

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

        # 3b. Presence filter (optional): drop edges that are zero for more than
        # (1 - threshold) of the subjects. Intended for sparse structural
        # connectomes (e.g. DTI streamline counts) where structural zeros should
        # not enter the model; leave off for functional data whose edges have a
        # real signed distribution around a mean of ~0. Uses X only (no target),
        # computed here on the training subjects, so it adds no leakage. This is
        # additive to the variance gate above, which already drops all-zero edges.
        presence_threshold = resolve_presence_threshold(self.presence_filter)
        if presence_threshold is not None:
            presence = (X != 0).float().mean(dim=0)
            valid_edges = valid_edges & (presence >= presence_threshold)

        if self.edge_statistic == 'pearson':
            r_edges_masked, p_edges_masked = correlations_and_pvalues(X=X, Y_perms=y,
                                                                      correlation_type='pearson')
        elif self.edge_statistic == 'spearman':
            r_edges_masked, p_edges_masked = correlations_and_pvalues(X=X, Y_perms=y,
                                                                      correlation_type='spearman')
        elif self.edge_statistic == 'pearson_partial':
            r_edges_masked, p_edges_masked = correlations_and_pvalues(X=X, Y_perms=y,
                                                                      confounds=covariates,
                                                                      correlation_type='pearson')
        elif self.edge_statistic == 'spearman_partial':
            r_edges_masked, p_edges_masked = correlations_and_pvalues(X=X, Y_perms=y,
                                                                      confounds=covariates,
                                                                      correlation_type='spearman')
        elif self.edge_statistic == 'point_biserial':
            # Point-biserial is Pearson against a binary 0/1 target; the unified
            # OLS path handles it with no special-casing.
            r_edges_masked, p_edges_masked = correlations_and_pvalues(X=X, Y_perms=y,
                                                                      correlation_type='pearson')
        elif self.edge_statistic == 'point_biserial_partial':
            r_edges_masked, p_edges_masked = correlations_and_pvalues(X=X, Y_perms=y,
                                                                      confounds=covariates,
                                                                      correlation_type='pearson')
        else:
            raise NotImplementedError("Unsupported edge selection method")
        # no dynamic shape change
        mask = valid_edges.to(r_edges_masked.dtype).unsqueeze(1)
        r_edges = r_edges_masked.to(r_edges.dtype) * mask
        p_edges = p_edges_masked.to(p_edges.dtype) * mask + (1.0 - mask)
        return r_edges, p_edges

    def fit_transform_batched(self, X, y, covariates, valid_mask, device):
        """
        Batched-over-folds version of `fit_transform` for zero-padded,
        ragged fold batches (see utils.build_fold_batch). Same
        edge_statistic dispatch as `fit_transform`, but operating on
        `[B, N_max, ...]` padded inputs and returning an extra folds
        dimension: `[N_features, B_folds, N_perms]` instead of
        `[N_features, N_perms]`.

        Args:
            X: [B, N_max, N_features] -- padded, samples axis = dim 1.
            y: [B, N_max, N_perms] -- padded.
            covariates: optional [B, N_max, N_cov] -- padded.
            valid_mask: [B, N_max] bool.
            device: torch device.

        Returns:
            r_edges, p_edges: [N_features, B_folds, N_perms].
        """
        X = torch.as_tensor(X, device=device, dtype=torch.float32)
        y = torch.as_tensor(y, device=device, dtype=torch.float32)
        valid_mask = torch.as_tensor(valid_mask, device=device, dtype=torch.bool)
        if covariates is not None:
            covariates = torch.as_tensor(covariates, device=device, dtype=torch.float32)

        if self.edge_statistic not in _EDGE_STATISTIC_DISPATCH:
            raise NotImplementedError("Unsupported edge selection method")
        correlation_type, use_confounds = _EDGE_STATISTIC_DISPATCH[self.edge_statistic]

        # Variance threshold (masked, per fold): drop near-constant edges to
        # avoid NaNs, mirroring fit_transform's torch.var(X, dim=0) but
        # restricted to each fold's real (valid) rows.
        mask = valid_mask.to(X.dtype).unsqueeze(-1)                        # [B, N_max, 1]
        n_valid = valid_mask.sum(dim=1).to(X.dtype)                        # [B]
        X_mean = (X * mask).sum(dim=1, keepdim=True) / n_valid.view(-1, 1, 1)
        X_var = ((X - X_mean) * mask).pow(2).sum(dim=1) / (n_valid.view(-1, 1) - 1).clamp(min=1)
        valid_edges = X_var > 1e-6                                          # [B, N_features]

        # Presence filter (optional, see fit_transform): masked mean over each
        # fold's real (valid) rows only, so padding rows don't bias it down.
        presence_threshold = resolve_presence_threshold(self.presence_filter)
        if presence_threshold is not None:
            presence = ((X != 0).to(X.dtype) * mask).sum(dim=1) / n_valid.view(-1, 1)  # [B, N_features]
            valid_edges = valid_edges & (presence >= presence_threshold)

        r_edges_masked, p_edges_masked = correlations_and_pvalues_batched(
            X=X, Y_perms=y, valid_mask=valid_mask, correlation_type=correlation_type,
            confounds=covariates if use_confounds else None)

        # r_edges_masked/p_edges_masked: [N_features, B_folds, N_perms]
        feat_mask = valid_edges.permute(1, 0).unsqueeze(-1).to(r_edges_masked.dtype)  # [F, B, 1]
        r_edges = r_edges_masked * feat_mask
        p_edges = p_edges_masked * feat_mask + (1.0 - feat_mask)
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
    presence_filter: bool or float, default=False
        Optional pre-filter that keeps only edges which are nonzero in at least a
        given fraction of subjects, dropping structural/near-zero edges before
        selection. ``True`` uses a fraction of ``0.5`` (present in the majority);
        a float sets the fraction explicitly (e.g. ``0.75``). Intended for sparse
        structural connectomes (e.g. DTI streamline counts); leave off
        (``False``) for functional data, whose edges have a real signed
        distribution around a mean of ~0. Computed per fold on the training
        subjects from the connectome only, so it adds no target leakage. Note:
        with ``CPMAnalysis(calculate_residuals=True)`` the connectome is
        residualized before selection, so the filter then sees residualized (not
        raw) values.
    connected_components: bool or int, default=False
        If set, keep only selected edges that belong to a connected component
        with at least this many edges (per positive/negative network), dropping
        isolated edges — this can improve edge stability. ``True`` uses a minimum
        of ``2`` edges (drop lone single edges); an int sets the minimum
        explicitly. Applied per fold and per permutation after thresholding.
    edge_selection: list of selectors (e.g. PThreshold), default=None
        One or more selection strategies. Provide a list of multiple
        configurations to tune them via an inner CV.
    """
    def __init__(self,
                 edge_statistic: str = 'spearman',
                 presence_filter: Union[bool, float] = False,
                 connected_components: Union[bool, int] = False,
                 edge_selection: Union[list, None, PThreshold] = None):
        self.r_edges = None
        self.p_edges = None
        self.presence_filter = presence_filter
        self.connected_components = connected_components
        self.edge_statistic = EdgeStatistic(edge_statistic=edge_statistic,
                                            presence_filter=presence_filter)
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
        min_edges = resolve_min_component_size(self.connected_components)
        if min_edges is not None:
            selected_edges = filter_connected_components(selected_edges, min_edges)
        return selected_edges
