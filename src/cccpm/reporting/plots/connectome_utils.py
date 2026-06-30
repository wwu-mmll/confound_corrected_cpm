"""
Connectome / atlas utilities for the brain-figure module.

These helpers are deliberately self-contained (they operate on plain numpy
matrices and a pandas atlas table) so the whole brain-figure layer can later be
lifted into the shared ``wwu-mmll/brainplots`` toolbox with minimal change.

The network colour scheme and ``upper_tri_to_matrix`` are adapted from that
toolbox's ``brainchord`` package (Juri Howe) to keep the two consistent.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Network colours (Yeo-7 / Schaefer / HCP inspired) ─────────────────────────
# Adapted from brainchord.atlas.DEFAULT_NETWORK_COLORS.
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


# ── Matrix helpers ────────────────────────────────────────────────────────────

def upper_tri_to_matrix(n_nodes: int, edge_values: np.ndarray) -> np.ndarray:
    """Reconstruct a symmetric ``n_nodes × n_nodes`` matrix from an upper-tri vector."""
    mat = np.zeros((n_nodes, n_nodes))
    rows, cols = np.triu_indices(n_nodes, k=1)
    mat[rows, cols] = edge_values
    mat[cols, rows] = edge_values
    return mat


def signed_stability_matrix(edge_stability: np.ndarray) -> np.ndarray:
    """
    Collapse a ``[n_nodes, n_nodes, 2(, runs)]`` stability array into one signed
    matrix: positive-network stability as ``+value``, negative as ``-value``.

    An edge belongs to at most one network, so the two layers do not overlap.
    """
    arr = np.asarray(edge_stability, dtype=float)
    # squeeze any trailing run dimension(s) down to [n, n, 2]
    while arr.ndim > 3:
        arr = arr[..., 0]
    pos = arr[:, :, 0]
    neg = arr[:, :, 1]
    return pos - neg


def significant_edge_matrices(
    edge_stability: np.ndarray,
    edge_significance: np.ndarray | None = None,
    *,
    alpha: float = 0.05,
    fallback_to_stable: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build binary positive/negative node×node matrices of the edges to highlight
    on the glass brain.

    An edge is kept if it was selected (stability > 0) and, when significance is
    available, its stability p-value is below *alpha*. If significance filtering
    leaves a network empty but stable edges exist, fall back to all stable edges
    (so the figure still renders — e.g. with few permutations the p-values can
    sit just above the threshold). Returns ``(positive_matrix, negative_matrix)``.
    """
    stab = np.asarray(edge_stability, dtype=float)
    while stab.ndim > 3:
        stab = stab[..., 0]  # drop trailing run dim(s) -> [n, n, 2]

    sig = None
    if edge_significance is not None:
        sig = np.asarray(edge_significance, dtype=float)
        while sig.ndim > 3:
            sig = sig[..., 0]

    out = []
    for layer in (0, 1):
        stable = stab[:, :, layer] > 1e-9
        keep = stable
        if sig is not None:
            keep = stable & (sig[:, :, layer] < alpha)
            if fallback_to_stable and not keep.any() and stable.any():
                keep = stable
        out.append(np.where(keep, 1.0, 0.0))
    return out[0], out[1]


def aggregate_matrix_by_group(
    matrix: np.ndarray,
    groups: list[str],
    *,
    signed: bool = True,
) -> pd.DataFrame:
    """
    Aggregate a node×node *matrix* into a region×region DataFrame by averaging
    over the off-diagonal node pairs that fall in each (group_i, group_j) block.

    Parameters
    ----------
    matrix : np.ndarray, shape (n_nodes, n_nodes)
        Symmetric node-level matrix (e.g. signed stability).
    groups : list[str], length n_nodes
        Group/network label for each node.
    signed : bool
        If False, aggregate absolute values.
    """
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
