"""Shared figure dimensions, colour palette, matplotlib style, and SVG save helper for the HTML report."""

from __future__ import annotations

import os

import matplotlib as mpl
import seaborn as sns


# Matches the brainchord toolbox hex codes so figures stay consistent with that package.
POS = "#d32f2f"
NEG = "#1565c0"
BOTH = "#9e9e9e"

COLOR_MAP = {
    "positive": POS,
    "negative": NEG,
    "both": BOTH,
}

DIVERGING_CMAP = "RdBu_r"

# Fixed vocabulary of sizes so report figure cards line up in the layout grid.
SQUARE = (3.2, 3.2)
PANEL = (3.2, 2.6)
WIDE = (6.6, 2.6)
WIDE_TALL = (6.6, 3.6)
HALF = (2.6, 2.6)


def apply_nature_style() -> None:
    """Apply the compact 'nature style' rcParams used by all report figures."""
    sns.set_theme(style="white")
    mpl.rcParams.update({
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "lines.linewidth": 0.75,
        "axes.linewidth": 0.5,
        "legend.fontsize": 6,
        "svg.fonttype": "none",  # keep text as text in the SVG (smaller, selectable)
    })


def save_figure(
    fig,
    path_no_ext: str,
    formats: tuple[str, ...] = ("svg", "pdf"),
    dpi: int = 600,
) -> str:
    """Save *fig* to ``path_no_ext`` in each requested format; returns the SVG path (or first format if no SVG)."""
    primary = None
    for fmt in formats:
        out = f"{path_no_ext}.{fmt}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
        if fmt == "svg":
            primary = out
        elif primary is None:
            primary = out
    return primary if primary is not None else f"{path_no_ext}.{formats[0]}"


def figure_path(results_folder: str, name: str) -> str:
    """Return the extension-less path for a figure inside *results_folder*."""
    return os.path.join(results_folder, name)
