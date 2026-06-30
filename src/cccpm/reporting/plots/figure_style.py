"""
Shared figure-style system for the CCCPM HTML report.

Single source of truth for figure dimensions, the colour palette, the
"nature style" matplotlib rcParams, and a save helper that emits vector SVG
(the format the Jinja2 report embeds inline).

Keeping all of this in one place is what makes the report's figures look
consistent: every figure is generated at one of a small set of standard sizes
and saved as crisp SVG, rather than at an arbitrary ``figsize`` and 600-DPI PNG.
"""

from __future__ import annotations

import os

import matplotlib as mpl
import seaborn as sns


# ── Colour palette ────────────────────────────────────────────────────────────
# One semantic palette used everywhere: positive network = red, negative = blue,
# both = grey. Matches the brainchord toolbox (#d32f2f / #1565c0) so figures are
# consistent with the future shared brain-plot package.
POS = "#d32f2f"   # positive network / positive correlation
NEG = "#1565c0"   # negative network / negative correlation
BOTH = "#9e9e9e"  # both / combined

COLOR_MAP = {
    "positive": POS,
    "negative": NEG,
    "both": BOTH,
}

# Diverging colormap for connectivity matrices (blue → white → red).
DIVERGING_CMAP = "RdBu_r"


# ── Standard figure sizes (inches) ────────────────────────────────────────────
# A small, fixed vocabulary of sizes. Every report figure picks one of these so
# the rendered cards line up in the layout grid.
SQUARE = (3.2, 3.2)     # scatter / matrix / chord — 1:1
PANEL = (3.2, 2.6)      # single boxplot / bar panel
WIDE = (6.6, 2.6)       # faceted small-multiples (metrics across)
WIDE_TALL = (6.6, 3.6)  # faceted grid with two rows
HALF = (2.6, 2.6)       # compact supporting figure


# ── Style ─────────────────────────────────────────────────────────────────────

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


# ── Saving ────────────────────────────────────────────────────────────────────

def save_figure(
    fig,
    path_no_ext: str,
    formats: tuple[str, ...] = ("svg", "pdf"),
    dpi: int = 600,
) -> str:
    """
    Save *fig* to ``path_no_ext`` in each requested format.

    SVG is the primary report format (embedded inline); PDF is kept for users
    who want a vector copy for papers. Returns the path to the SVG file (the one
    the report embeds), or the first written format if SVG was not requested.
    """
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
