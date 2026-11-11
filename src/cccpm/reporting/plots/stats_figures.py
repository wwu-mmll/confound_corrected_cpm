"""Statistics figures for the CCCPM HTML report (permutation null-distribution plots)."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from cccpm.reporting.plots.figure_style import (
    COLOR_MAP,
    PANEL,
    apply_nature_style,
    figure_path,
    save_figure,
)


def null_distribution_plot(
    max_null: Sequence[float],
    results_folder: str,
    name: str,
    *,
    network: str = "positive",
    observed: float | None = None,
    critical_value: float | None = None,
    statistic_label: str = "Max statistic",
) -> str:
    """Plot the permutation max-statistic (FWER) null distribution for one network.

    max_null is the per-permutation max component size (NBS) or max TFCE score (TFCE).
    """
    apply_nature_style()
    color = COLOR_MAP.get(network, COLOR_MAP["both"])
    null = np.asarray(list(max_null), dtype=float)

    fig, ax = plt.subplots(figsize=PANEL)
    if null.size:
        n_bins = int(min(40, max(10, np.unique(null).size)))
        ax.hist(null, bins=n_bins, color=color, alpha=0.55,
                edgecolor="white", linewidth=0.3)

    if critical_value is not None:
        ax.axvline(critical_value, color="#555555", linestyle="--", linewidth=0.9,
                   label=f"critical value ({critical_value:.3g})")
    if observed is not None:
        ax.axvline(observed, color=color, linestyle="-", linewidth=1.3,
                   label=f"observed ({observed:.3g})")

    ax.set_xlabel(statistic_label)
    ax.set_ylabel("Permutations")
    if critical_value is not None or observed is not None:
        ax.legend(frameon=False, loc="upper right")

    path_no_ext = figure_path(results_folder, name)
    out = save_figure(fig, path_no_ext)
    plt.close(fig)
    return out
