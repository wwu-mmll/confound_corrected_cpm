"""
Confound-inflation demonstration for CPM.

Shows, on data with an *analytically known* answer, that a connectome model's
apparent predictive performance is inflated by a common-cause confound, and that
**partial-correlation edge selection does not fix this** — only residualizing the
connectome recovers the true brain-behaviour association.

Data come from ``simulate_data_given_kappa`` (``cccpm.simulation.simulate_sem``):
a latent confound ``Z`` is a common cause of both ``X`` and ``y``. Two knobs are
swept: brain-outcome strength ``R2(y~X) ∈ {0.09, 0.36, 0.81}`` and spurious
fraction ``κ ∈ {0, 0.2, …, 1.0}`` (the confound-driven share of the naive R²),
giving a closed-form true value ``R2(y~X|Z) = (1 − κ)·R2(y~X)``.

Each connectome has three edge classes (10 each): pure-signal (related to y,
uncorrelated with the confound), confound-only (related to y only through the
confound), and mixed (both). This is what exposes partial selection's limit: it
rejects confound-only edges but keeps mixed edges, whose raw values still leak
confound variance into the network sum score — so partial selection reduces but
does not eliminate inflation, landing *between* raw CPM and the true value; only
residualization recovers the truth.

Each cell runs the 2x2 of ``selection_input`` x ``model_input`` ('raw' vs
'residualized') — i.e. raw, partial selection, residualized-X, and the fourth
cell (raw selection + residualized model, the old "residuals" model) — and
compares connectome out-of-sample explained variance to the naive and true
reference values. Expected result: raw ≈ partial ≈ R2(y~X) (flat, inflated),
while residualized-X and the residuals model track the falling true curve. The
widening gap is the headline.

Run:  poetry run python scripts/confound_inflation_demo.py
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
from cccpm.edge_selection import PThreshold, UnivariateEdgeSelection
from cccpm.simulation.simulate_sem import (
    simulate_data_given_kappa,
    compute_r2s,
    GRID_R2_X_Y,
    GRID_KAPPA,
)
from cccpm.validation import check_data

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
# CPMAnalysis.run() writes an HTML report per call and only exposes CV metrics,
# so this sweep calls the internal `_single_run` directly and stubs out tqdm
# (an object, not a lambda, since tqdm is driven via .update()/.close()).
class _SilentBar:
    """Stand-in for tqdm's progress bar that does nothing."""

    def __init__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def __iter__(self):
        return iter(())


_ca.tqdm = _SilentBar

R2_TARGETS = GRID_R2_X_Y
KAPPAS = GRID_KAPPA
N_SIMS = 10
N_SAMPLES = 2000
N_FEATURES = 105
N_MIXED = 10
N_PURE_SIGNAL = 10
N_CONFOUND_ONLY = 10
N_CONFOUNDS = 2
P_THRESHOLD = 0.05

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "confound_inflation")
# Per-run CPM artifacts are throwaway; kept out of RESULTS_DIR.
_SCRATCH = tempfile.mkdtemp(prefix="cccpm_confound_demo_")


def _edge_selection(selection_input):
    """Build a UnivariateEdgeSelection with a fixed p-threshold."""
    return UnivariateEdgeSelection(
        selection_statistic="pearson",
        selection_input=selection_input,
        edge_selection=[PThreshold(threshold=P_THRESHOLD, correction=[None])],
    )


def run_cpm(X, y, Z, selection_input, model_input="raw"):
    """Run one cell of the confound 2x2; return explained variance per model."""
    cpm = CPMAnalysis(
        results_directory=_SCRATCH,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        edge_selection=_edge_selection(selection_input),
        model_input=model_input,
        n_permutations=0,
        task_type="regression",
    )
    Xc, yc, Zc = check_data(X, y, Z, impute_missings=True)
    cpm._single_run(X=Xc, y=yc.reshape(-1, 1), covariates=Zc, perm_run=False)
    ag = cpm.results_manager.agg_results

    def ev(model):
        # agg_results keeps a runs level, so this is a one-element Series.
        return float(np.ravel(
            ag.loc[model, "both"]["explained_variance_score"]["mean"])[0])

    return {m: ev(m) for m in ("connectome", "covariates", "full")}


def selected_edge_mask(X, y, Z, selection_input):
    """Select edges on the full dataset; return a boolean (n_features,) mask of kept edges."""
    ue = _edge_selection(selection_input)
    ue.fit_transform(X=X, y=y.reshape(-1, 1), covariates=Z)
    selector = PThreshold(threshold=P_THRESHOLD, correction=None)
    edges = selector.select(r=ue.r_edges, t=ue.t_edges, df=ue.df)   # [F, 2, runs]
    return edges.bool().any(dim=1).squeeze(-1).cpu().numpy().astype(bool)


def run_sweep():
    """Sweep R2/kappa cells across all four confound-control configurations."""
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
                        n_features_informative=N_MIXED,
                        n_pure_signal_features=N_PURE_SIGNAL,
                        n_confound_only_features=N_CONFOUND_ONLY,
                        n_confounds=N_CONFOUNDS,
                        n_samples=N_SAMPLES,
                        random_state=seed,
                    )
                X, y, Z, info = sim["X"], sim["y"], sim["Z"], sim["info"]
                emp = compute_r2s(sim)

                raw = run_cpm(X, y, Z, "raw", "raw")
                partial = run_cpm(X, y, Z, "residualized", "raw")
                residx = run_cpm(X, y, Z, "residualized", "residualized")
                raw_resid = run_cpm(X, y, Z, "raw", "residualized")

                records.append(dict(
                    r2_target=r2, kappa=kappa, sim=sim_idx,
                    true_r2=true_r2,
                    emp_naive=emp["r2_naive"], emp_unique_x=emp["r2_unique_X"],
                    connectome_raw=raw["connectome"],
                    connectome_partial=partial["connectome"],
                    connectome_residualizedX=residx["connectome"],
                    residuals_model=raw_resid["connectome"],
                    increment_full=raw["full"] - raw["covariates"],
                ))

                classes = {
                    "pure_signal": info["pure_signal_idx"],
                    "mixed": info["mixed_idx"],
                    "confound_only": info["confound_only_idx"],
                }
                for selection_input, label in (("raw", "raw"),
                                               ("residualized", "partial")):
                    m = selected_edge_mask(X, y, Z, selection_input)
                    edge_records.append(dict(
                        r2_target=r2, kappa=kappa, sim=sim_idx, selection=label,
                        n_pure_signal=int(m[classes["pure_signal"]].sum()),
                        n_mixed=int(m[classes["mixed"]].sum()),
                        n_confound_only=int(m[classes["confound_only"]].sum()),
                        n_total=int(m.sum()),
                    ))

    return pd.DataFrame(records), pd.DataFrame(edge_records)


BAR_METHODS = [
    ("connectome_raw", "Raw", "#4C72B0"),
    ("connectome_partial", "Partial selection", "#DD8452"),
    ("connectome_residualizedX", "Residualized X", "#55A868"),
    ("residuals_model", "Residuals model", "#8172B3"),
]


def plot_inflation(df, path):
    """Plot mean connectome R² vs kappa per method, one panel per R² target."""
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
    """Plot mean edges selected per class (pure-signal/mixed/confound-only) vs kappa."""
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


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Running confound-inflation sweep "
          f"({len(R2_TARGETS)}×{len(KAPPAS)} cells × {N_SIMS} sims)...")
    df, edf = run_sweep()

    csv_path = os.path.join(RESULTS_DIR, "confound_inflation_results.csv")
    df.to_csv(csv_path, index=False)
    edf.to_csv(os.path.join(RESULTS_DIR, "edge_selection_counts.csv"), index=False)

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
