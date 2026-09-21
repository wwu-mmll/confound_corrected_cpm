"""
Memory planning for the permutation loop.

Permutations are always vectorised -- see `plan_permutation_chunk` for why this
only caps how many columns are in flight and never falls back to a loop.
"""
import torch


def available_memory_bytes(device) -> int:
    """Free memory on `device`: CUDA via mem_get_info, CPU via psutil."""
    device = torch.device(device)
    if device.type == 'cuda':
        free_bytes, _total = torch.cuda.mem_get_info(device)
        return int(free_bytes)
    import psutil
    return int(psutil.virtual_memory().available)


def plan_permutation_chunk(n_features, n_samples, n_runs, device,
                           safety_factor: float = 0.4) -> int:
    """
    How many permutation columns can be processed at once without running out
    of memory.

    Permutations are always *vectorised* -- y is [N_samples, N_runs] and the
    edge statistic is one matrix product over all columns, which is where the
    GPU speedup comes from (measured 433x versus looping over columns). This
    function does not change that; it only caps how many columns are in flight,
    so a large parcellation crossed with many permutations degrades in speed
    instead of crashing.

    The per-column cost is dominated by the ~10 [N_features, N_runs] float32
    temporaries inside `correlations_and_pvalues` (residualised X and y, cross
    products, r, t, z, p, ...). Measured directly at ~40 bytes per
    (run x feature), stable across n_features from 4,950 to 35,778; the
    n_samples term covers the prediction and rank tensors, which are much
    smaller.

    The estimate is deliberately conservative (`safety_factor` keeps well clear
    of the free-memory figure). The predecessor of this function, the
    `batch_planning.estimate_bytes` cost model, under-counted -- it modelled
    neither the imputation intermediates nor the O(N^2) ROC-AUC term -- and
    planned batches that then OOMed. Under-counting is worse than not
    estimating at all, so this errs low and accepts extra chunks.

    Returns a chunk size in [1, n_runs].
    """
    bytes_per_run = 40 * n_features + 240 * n_samples
    budget = available_memory_bytes(device) * safety_factor
    chunk = int(budget // max(bytes_per_run, 1))
    return max(1, min(chunk, n_runs))
