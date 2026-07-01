"""
Confound-inflation demonstration for CPM.

Shows, on data with an *analytically known* answer, that a connectome model's
apparent predictive performance is inflated by a common-cause confound, and that
**partial-correlation edge selection does not fix this** — only residualizing the
connectome (or otherwise controlling the confound in the model) recovers the true
brain–behaviour association.

Design
------
The data are generated with ``simulate_data_given_kappa`` (see
``cccpm.simulation.simulate_sem``). A single latent confound ``Z`` is a common
cause of both the connectome ``X`` and the outcome ``y``. Two knobs are swept:

* rows  — brain–outcome strength   ``R2(y~X) ∈ {0.09, 0.36, 0.81}``
* cols  — spurious fraction ``κ ∈ {0, 0.2, …, 1.0}`` — the fraction of the naive
          R² that is confound-driven. The true, deconfounded value is known in
          closed form: ``R2(y~X|Z) = (1 − κ)·R2(y~X)``.

Each connectome contains three interpretable edge classes (10 each by default):

* **pure-signal** edges — related to y but uncorrelated with the confound,
* **confound-only** edges — related to y only through the confound,
* **mixed** edges — carrying *both* genuine signal and confound variance.

This separation is what exposes the limit of partial-correlation selection: it
rejects confound-only edges (their partial association with y is null) but keeps
mixed edges (which retain a genuine partial association), and the raw values of
those mixed edges still leak confound variance into the network sum score. Hence
partial selection reduces, but does not eliminate, the inflation — the connectome
model lands **between** raw CPM and the true value; only residualization recovers
the truth.

For every cell the toolbox is run in three configurations and the connectome
model's out-of-sample explained variance is compared to the naive and the true
reference values:

    (a) raw            — edge_statistic='pearson',         calculate_residuals=False
    (b) partial        — edge_statistic='pearson_partial',  calculate_residuals=False
    (c) residualized-X — edge_statistic='pearson',          calculate_residuals=True

We also read the toolbox's own ``residuals`` model (residualizes the aggregate
strength) as a fourth, alternative deconfounding route.

Expected result: raw ≈ partial ≈ R2(y~X) (flat across κ, inflated), while
residualized-X and the residuals model track the falling true curve
(1 − κ)·R2(y~X). The widening gap is the headline.

Run:  poetry run python examples/confound_inflation_demo.py
"""

import contextlib
import logging
import os
import shutil
import tempfile
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from cccpm import CPMAnalysis
import cccpm.cpm_analysis as _ca
from cccpm.constants import TaskType
from cccpm.edge_selection import PThreshold, UnivariateEdgeSelection
from cccpm.simulation.simulate_sem import (
    simulate_data_given_kappa,
    compute_r2s,
    GRID_R2_X_Y,
    GRID_KAPPA,
)
from cccpm.utils import check_data

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
# The full CPMAnalysis.run() also writes an HTML report per call (~seconds); we
# only need the cross-validated metrics, so we call the internal _single_run and
# silence the per-fold progress bar for this batch sweep.
_ca.tqdm = lambda iterable, **kwargs: iterable

# ── Configuration ───────────────────────────────────────────────────────────
R2_TARGETS = GRID_R2_X_Y            # (0.09, 0.36, 0.81)
KAPPAS = GRID_KAPPA                 # (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
N_SIMS = 10
N_SAMPLES = 2000
N_FEATURES = 105                    # 15-node connectome
N_MIXED = 10                        # mixed edges (signal + confound)
N_PURE_SIGNAL = 10                  # pure-signal edges (⟂ confound)
N_CONFOUND_ONLY = 10                # confound-only edges
N_CONFOUNDS = 2
P_THRESHOLD = 0.05

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "confound_inflation")
# Per-run CPM artifacts (edges, predictions, ...) are throwaway; keep them out of
# RESULTS_DIR so only the figures and CSVs land there.
_SCRATCH = tempfile.mkdtemp(prefix="cccpm_confound_demo_")


# ── Toolbox helpers ─────────────────────────────────────────────────────────
def _edge_selection(statistic):
    return UnivariateEdgeSelection(
        edge_statistic=statistic,
        edge_selection=[PThreshold(threshold=P_THRESHOLD, correction=[None])],
    )


def run_cpm(X, y, Z, statistic, calculate_residuals):
    """Run one CPM configuration and return the 'both'-network explained variance
    for every model variant (connectome/covariates/full/residuals)."""
    cpm = CPMAnalysis(
        results_directory=_SCRATCH,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        edge_selection=_edge_selection(statistic),
        calculate_residuals=calculate_residuals,
        n_permutations=0,
        task_type="regression",
    )
    Xc, yc, Zc = check_data(X, y, Z, impute_missings=True)
    cpm._single_run(X=Xc, y=yc.reshape(-1, 1), covariates=Zc, perm_run=False)
    ag = cpm.results_manager.agg_results

    def ev(model):
        return float(ag.loc[model, "both"]["explained_variance_score"]["mean"])

    return {m: ev(m) for m in ("connectome", "covariates", "full", "residuals")}


def selected_edge_mask(X, y, Z, statistic):
    """Boolean mask (n_features,) of edges selected on the full dataset."""
    ue = _edge_selection(statistic)
    ue.fit_transform(X=X, y=y.reshape(-1, 1), covariates=Z)
    selector = PThreshold(threshold=P_THRESHOLD, correction=None)
    edges = selector.select(r=ue.r_edges, p=ue.p_edges)   # [F, 2, runs]
    return edges.bool().any(dim=1).squeeze(-1).cpu().numpy().astype(bool)


# ── Sweep ───────────────────────────────────────────────────────────────────
def run_sweep():
    rng = np.random.default_rng(2024)
    records = []
    edge_records = []

    for r2 in R2_TARGETS:
        for kappa in KAPPAS:
            true_r2 = (1.0 - kappa) * r2
            for sim_idx in range(N_SIMS):
                seed = int(rng.integers(0, 2**32 - 1))
                # κ=1 clamps rho→0.99 and prints a benign "closest value" notice; mute it.
                with contextlib.redirect_stdout(open(os.devnull, "w")):
                    sim = simulate_data_given_kappa(
                        R2_X_y=r2, kappa=kappa,
                        n_features=N_FEATURES,
                        n_features_informative=N_MIXED,          # mixed edges
                        n_pure_signal_features=N_PURE_SIGNAL,    # pure-signal edges
                        n_confound_only_features=N_CONFOUND_ONLY,
                        n_confounds=N_CONFOUNDS,
                        n_samples=N_SAMPLES,
                        random_state=seed,
                    )
                X, y, Z, info = sim["X"], sim["y"], sim["Z"], sim["info"]

                # Validate the generated data matches the analytic targets.
                emp = compute_r2s(sim)

                raw = run_cpm(X, y, Z, "pearson", False)
                partial = run_cpm(X, y, Z, "pearson_partial", False)
                residx = run_cpm(X, y, Z, "pearson", True)

                records.append(dict(
                    r2_target=r2, kappa=kappa, sim=sim_idx,
                    true_r2=true_r2,
                    emp_naive=emp["r2_naive"], emp_unique_x=emp["r2_unique_X"],
                    connectome_raw=raw["connectome"],
                    connectome_partial=partial["connectome"],
                    connectome_residualizedX=residx["connectome"],
                    residuals_model=raw["residuals"],
                    increment_full=raw["full"] - raw["covariates"],
                ))

                # Edge-selection composition (raw vs partial) on the full data.
                classes = {
                    "pure_signal": info["pure_signal_idx"],
                    "mixed": info["mixed_idx"],
                    "confound_only": info["confound_only_idx"],
                }
                for stat, label in (("pearson", "raw"), ("pearson_partial", "partial")):
                    m = selected_edge_mask(X, y, Z, stat)
                    edge_records.append(dict(
                        r2_target=r2, kappa=kappa, sim=sim_idx, selection=label,
                        n_pure_signal=int(m[classes["pure_signal"]].sum()),
                        n_mixed=int(m[classes["mixed"]].sum()),
                        n_confound_only=int(m[classes["confound_only"]].sum()),
                        n_total=int(m.sum()),
                    ))

    return pd.DataFrame(records), pd.DataFrame(edge_records)


# ── Plotting ────────────────────────────────────────────────────────────────
BAR_METHODS = [
    ("connectome_raw", "Raw", "#4C72B0"),
    ("connectome_partial", "Partial selection", "#DD8452"),
    ("connectome_residualizedX", "Residualized X", "#55A868"),
    ("residuals_model", "Residuals model", "#8172B3"),
]


def plot_inflation(df, path):
    agg = (df.groupby(["r2_target", "kappa"])
             .agg(["mean", "std"]))
    fig, axes = plt.subplots(1, len(R2_TARGETS), figsize=(16, 5), sharey=False)

    for ax, r2 in zip(axes, R2_TARGETS):
        sub = agg.xs(r2, level="r2_target").reset_index().sort_values("kappa")
        x = np.arange(len(sub))
        n = len(BAR_METHODS)
        width = 0.8 / n
        for i, (col, label, color) in enumerate(BAR_METHODS):
            ax.bar(x + (i - (n - 1) / 2) * width, sub[(col, "mean")],
                   width=width, color=color, label=label,
                   yerr=sub[(col, "std")], capsize=2,
                   error_kw=dict(elinewidth=0.8, alpha=0.6))
        # Reference lines: naive (inflated) and true (deconfounded).
        ax.axhline(r2, color="black", lw=1.5, ls="-", label="Naive R²(y~X)")
        ax.plot(x, sub[("true_r2", "mean")], "r--o", lw=1.5, ms=4,
                label="True R²(y~X|Z)")
        ax.set_title(f"R²(y~X) = {r2}", fontweight="bold")
        ax.set_xlabel("Spurious fraction κ")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{k:.1f}" for k in sub["kappa"]])
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("Connectome prediction R² (explained variance)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        f"Confound inflation in CPM — mean ± SD over {N_SIMS} simulations\n"
        "Raw is maximally inflated; partial-correlation selection only partially "
        "deconfounds; residualization recovers the truth",
        y=1.03, fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               ncol=6, fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


EDGE_CLASSES = [
    ("n_pure_signal", "pure-signal", "#4C72B0", "s"),
    ("n_mixed", "mixed", "#8172B3", "^"),
    ("n_confound_only", "confound-only", "#C44E52", "o"),
]


def plot_edge_selection(edf, path):
    cols = [c for c, *_ in EDGE_CLASSES]
    agg = (edf.groupby(["r2_target", "kappa", "selection"])
              [cols].mean().reset_index())
    fig, axes = plt.subplots(1, len(R2_TARGETS), figsize=(16, 4.5), sharey=True)
    for ax, r2 in zip(axes, R2_TARGETS):
        sub = agg[agg["r2_target"] == r2]
        for sel, ls in (("raw", "-"), ("partial", "--")):
            s = sub[sub["selection"] == sel].sort_values("kappa")
            for col, cname, color, marker in EDGE_CLASSES:
                ax.plot(s["kappa"], s[col], ls, color=color, marker=marker,
                        label=f"{cname} ({sel})")
        ax.set_title(f"R²(y~X) = {r2}", fontweight="bold")
        ax.set_xlabel("Spurious fraction κ")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean # edges selected")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        "Edge selection by class: partial selection rejects confound-only edges "
        "but keeps mixed edges (whose raw values still leak confound)",
        y=1.02, fontsize=12)
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=3, fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Running confound-inflation sweep "
          f"({len(R2_TARGETS)}×{len(KAPPAS)} cells × {N_SIMS} sims)...")
    df, edf = run_sweep()

    csv_path = os.path.join(RESULTS_DIR, "confound_inflation_results.csv")
    df.to_csv(csv_path, index=False)
    edf.to_csv(os.path.join(RESULTS_DIR, "edge_selection_counts.csv"), index=False)

    # Summary table (mean over sims).
    summary = (df.groupby(["r2_target", "kappa"])[
        ["true_r2", "emp_naive", "connectome_raw", "connectome_partial",
         "connectome_residualizedX", "residuals_model"]].mean())
    pd.set_option("display.width", 160, "display.max_columns", 20)
    print("\nMean connectome R² over simulations "
          "(naive is the inflated reference, true_r2 the target):")
    print(summary.round(3).to_string())

    fig1 = os.path.join(RESULTS_DIR, "confound_inflation.pdf")
    fig2 = os.path.join(RESULTS_DIR, "edge_selection_composition.pdf")
    plot_inflation(df, fig1)
    plot_edge_selection(edf, fig2)
    print(f"\nSaved:\n  {csv_path}\n  {fig1}\n  {fig2}")

    shutil.rmtree(_SCRATCH, ignore_errors=True)


if __name__ == "__main__":
    main()
