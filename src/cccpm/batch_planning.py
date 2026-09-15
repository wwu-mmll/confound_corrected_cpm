"""
Memory-aware batch-size planning for the CPM outer/inner CV loops.

The pipeline has three loop dimensions that can be represented as tensor batch
dimensions instead of Python for-loops: hyperparameters ("params"), CV folds,
and permutations ("perms", already riding along as extra columns of `y`
everywhere in the codebase). This module decides, given how much GPU/CPU
memory is actually available, how large a batch to take along each of those
dimensions -- escalating in priority order params -> folds -> perms, and
falling back to a batch size of 1 (today's exact per-item loop) wherever
memory is tight.
"""
from dataclasses import dataclass
from typing import Callable

import torch

from cccpm.results_manager import ResultsManager


@dataclass(frozen=True)
class BatchPlan:
    params: int
    folds: int
    perms: int


def available_memory_bytes(device) -> int:
    """Free memory on `device`: CUDA via mem_get_info, CPU via psutil."""
    device = torch.device(device)
    if device.type == 'cuda':
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
        return int(free_bytes)

    import psutil
    return int(psutil.virtual_memory().available)


def estimate_bytes(n_samples_train, n_samples_test, n_features, n_cov,
                    b_params, b_folds, b_perms,
                    dtype_bytes=4, overhead_multiplier=3.0):
    """
    Conservative upper-bound estimate of the peak bytes needed to process one
    (b_params, b_folds, b_perms)-sized batch through edge selection, model
    fit/predict, scoring, and results storage.

    This does not enumerate every transient torch intermediate individually
    (mean-centred copies, cross products, IRLS working arrays, ...) -- it
    covers the tensors that actually scale with the batch dimensions, then
    applies `overhead_multiplier` to account for everything else. The caller
    additionally applies a `safety_factor` to the available-memory side (see
    `plan_batch_sizes`), so this estimate does not need to be exact, only
    monotonically increasing in each batch dimension.
    """
    # Padded fold-batch copies of X/y/covariates (train + test).
    fold_batch_bytes = b_folds * dtype_bytes * (
        n_samples_train * n_features       # X_train
        + n_samples_train * b_perms        # y_train
        + n_samples_train * n_cov          # covariates_train
        + n_samples_test * n_features      # X_test
        + n_samples_test * b_perms         # y_test
        + n_samples_test * n_cov           # covariates_test
    )

    # Edge-statistic r/p tensors: [N_features, b_folds, b_perms], float32, x2 (r and p).
    edge_stat_bytes = 2 * n_features * b_folds * b_perms * dtype_bytes

    # Edge mask: [N_features, 2, b_params, b_folds, b_perms], bool (1 byte).
    edge_mask_bytes = 2 * n_features * b_params * b_folds * b_perms

    # Strength / solve intermediates: design width is small (~n_cov + 3),
    # scales with n_samples, not n_features.
    design_width = n_cov + 3
    model_bytes = dtype_bytes * b_params * b_folds * b_perms * design_width * (
        n_samples_train + n_samples_test
    )

    # Prediction tensor: [N_samples_test, N_models=5, N_networks=3, b_params, b_folds, b_perms].
    prediction_bytes = dtype_bytes * n_samples_test * 5 * 3 * b_params * b_folds * b_perms

    # results_manager slice being written this batch.
    results_bytes = ResultsManager.estimate_slice_bytes(
        n_features=n_features, n_params=b_params, n_folds=b_folds, n_runs=b_perms)

    total = (fold_batch_bytes + edge_stat_bytes + edge_mask_bytes
             + model_bytes + prediction_bytes + results_bytes)
    return int(total * overhead_multiplier)


def make_cpm_cost_fn(n_samples_train, n_samples_test, n_features, n_cov,
                      dtype_bytes=4, overhead_multiplier=3.0) -> Callable[[int, int, int], int]:
    """Bind the fixed problem-size args, returning a cost_fn(b_params, b_folds, b_perms)."""
    def cost_fn(b_params, b_folds, b_perms):
        return estimate_bytes(
            n_samples_train=n_samples_train, n_samples_test=n_samples_test,
            n_features=n_features, n_cov=n_cov,
            b_params=b_params, b_folds=b_folds, b_perms=b_perms,
            dtype_bytes=dtype_bytes, overhead_multiplier=overhead_multiplier)
    return cost_fn


def plan_batch_sizes(n_params: int, n_folds: int, n_perms: int,
                      cost_fn: Callable[[int, int, int], int],
                      available_bytes: int, safety_factor: float = 0.8) -> BatchPlan:
    """
    Decide batch sizes for (params, folds, perms) that fit inside
    `available_bytes * safety_factor`, escalating in priority order
    params -> folds -> perms.

    Starts from (1, 1, 1) and grows each dimension in turn to the largest size that still
    fits, holding already-decided dimensions fixed and not-yet-visited ones
    at 1. Requires `cost_fn` to be monotonically non-decreasing in each of
    its three arguments (true of `estimate_bytes`).
    """
    budget = available_bytes * safety_factor
    totals = {'params': max(1, n_params), 'folds': max(1, n_folds), 'perms': max(1, n_perms)}
    sizes = {'params': 1, 'folds': 1, 'perms': 1}

    def fits(candidate):
        return cost_fn(candidate['params'], candidate['folds'], candidate['perms']) <= budget

    for dim in ('params', 'folds', 'perms'):
        lo, hi = 1, totals[dim]
        if hi <= 1:
            continue

        candidate = dict(sizes)
        candidate[dim] = hi
        if fits(candidate):
            sizes[dim] = hi
            continue

        best = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate[dim] = mid
            if fits(candidate):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        sizes[dim] = best

    return BatchPlan(params=sizes['params'], folds=sizes['folds'], perms=sizes['perms'])
