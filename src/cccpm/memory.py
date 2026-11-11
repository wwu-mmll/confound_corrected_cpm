"""Memory planning for the permutation loop (always vectorised; see `plan_permutation_chunk`)."""
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
    """Caps how many permutation columns (of n_runs) are processed at once, so memory degrades to more chunks instead of OOM.

    bytes_per_run (40*n_features + 240*n_samples) comes from measuring the
    ~10 [N_features, N_runs] float32 temporaries inside `edge_statistics`
    (~40 bytes per run x feature, stable across n_features 4,950-35,778).
    Deliberately conservative: the predecessor cost model under-counted
    (missed imputation intermediates and the O(N^2) ROC-AUC term) and OOMed.
    """
    bytes_per_run = 40 * n_features + 240 * n_samples
    budget = available_memory_bytes(device) * safety_factor
    chunk = int(budget // max(bytes_per_run, 1))
    return max(1, min(chunk, n_runs))
