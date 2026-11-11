"""Connectome / atlas utilities for the brain-figure module.

Network colours and ``upper_tri_to_matrix`` are adapted from brainchord to keep the two consistent.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_NETWORK_COLORS: dict[str, str] = {
    "Vis":          "#781286",
    "SomMot":       "#4682b4",
    "DorsAttn":     "#00760e",
    "SalVentAttn":  "#c43afa",
    "Limbic":       "#dcf8a4",
    "Cont":         "#e69422",
    "Default":      "#cd3e4e",
    "Subcortical":  "#7b1fa2",
    "Thalamus":     "#6a1b9a",
    "Cerebellum":   "#1565c0",
    "Brainstem":    "#c2185b",
    "Prefrontal":   "#d32f2f",
    "Frontal":      "#e53935",
    "Motor":        "#c62828",
    "Premotor":     "#b71c1c",
    "Insula":       "#f9a825",
    "Parietal":     "#388e3c",
    "Temporal":     "#43a047",
    "Occipital":    "#00897b",
    "Cingulate":    "#0277bd",
}

_FALLBACK_CYCLE = list(plt.cm.tab20.colors)
_fallback_map: dict[str, str] = {}


def get_network_color(name: str, custom: dict[str, str] | None = None) -> str:
    """Return a hex colour for a network/region *name* (exact → fuzzy → fallback)."""
    lookup = {**DEFAULT_NETWORK_COLORS, **(custom or {})}
    if name in lookup:
        return lookup[name]
    lower = str(name).lower()
    for key, color in lookup.items():
        if key.lower() in lower or lower in key.lower():
            return color
    if name not in _fallback_map:
        idx = len(_fallback_map) % len(_FALLBACK_CYCLE)
        _fallback_map[name] = mcolors.to_hex(_FALLBACK_CYCLE[idx])
    return _fallback_map[name]


def upper_tri_to_matrix(n_nodes: int, edge_values: np.ndarray) -> np.ndarray:
    """Reconstruct a symmetric ``n_nodes × n_nodes`` matrix from an upper-tri vector."""
    mat = np.zeros((n_nodes, n_nodes))
    rows, cols = np.triu_indices(n_nodes, k=1)
    mat[rows, cols] = edge_values
    mat[cols, rows] = edge_values
    return mat


def signed_stability_matrix(edge_stability: np.ndarray) -> np.ndarray:
    """Collapse a ``[n_nodes, n_nodes, 2(, runs)]`` stability array into one signed (+pos/-neg) matrix."""
    arr = np.asarray(edge_stability, dtype=float)
    while arr.ndim > 3:
        arr = arr[..., 0]
    pos = arr[:, :, 0]
    neg = arr[:, :, 1]
    return pos - neg


def masked_signed_stability_matrix(
    edge_stability: np.ndarray,
    edge_significance: np.ndarray | None = None,
    *,
    mode: str = "significant",
    alpha: float = 0.05,
) -> np.ndarray:
    """Signed node x node stability matrix keeping only the edges selected by *mode*.

    mode: "all" keeps every stable edge; "significant" keeps p < alpha (falls back to
    all stable edges for a network if that would empty it, e.g. too few permutations);
    "top5"/"top10" keep the top 5%/10% by stability per network layer.
    """
    arr = np.asarray(edge_stability, dtype=float)
    while arr.ndim > 3:
        arr = arr[..., 0]

    sig = None
    if edge_significance is not None:
        sig = np.asarray(edge_significance, dtype=float)
        while sig.ndim > 3:
            sig = sig[..., 0]

    top_pct = {"top5": 5.0, "top10": 10.0}.get(mode)
    layers = []
    for layer in (0, 1):
        vals = arr[:, :, layer]
        stable = vals > 1e-9
        if mode == "significant" and sig is not None:
            keep = stable & (sig[:, :, layer] < alpha)
            if not keep.any() and stable.any():
                keep = stable
        elif top_pct is not None:
            keep = np.zeros_like(stable)
            stable_vals = vals[stable]
            if stable_vals.size:
                threshold = np.percentile(stable_vals, 100.0 - top_pct)
                keep = stable & (vals >= threshold)
        else:  # "all", or "significant" without significance data
            keep = stable
        layers.append(np.where(keep, vals, 0.0))
    return layers[0] - layers[1]


def aggregate_matrix_by_group(
    matrix: np.ndarray,
    groups: list[str],
    *,
    signed: bool = True,
) -> pd.DataFrame:
    """Aggregate a node x node *matrix* into a region x region DataFrame by averaging each (group_i, group_j) block."""
    n_nodes = matrix.shape[0]
    if len(groups) != n_nodes:
        raise ValueError(f"groups has {len(groups)} entries but matrix has {n_nodes} nodes.")

    vals = matrix if signed else np.abs(matrix)
    unique = list(dict.fromkeys(groups))
    idx = {g: i for i, g in enumerate(unique)}
    k = len(unique)

    sums = np.zeros((k, k))
    counts = np.zeros((k, k), dtype=int)
    rows, cols = np.triu_indices(n_nodes, k=1)
    for r, c in zip(rows, cols):
        gi, gj = idx[groups[r]], idx[groups[c]]
        sums[gi, gj] += vals[r, c]
        sums[gj, gi] += vals[r, c]
        counts[gi, gj] += 1
        counts[gj, gi] += 1

    with np.errstate(invalid="ignore"):
        mean = np.where(counts > 0, sums / counts, 0.0)
    return pd.DataFrame(mean, index=unique, columns=unique)
