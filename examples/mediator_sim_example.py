import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from cccpm import CPMAnalysis
from cccpm.edge_selection import PThreshold, UnivariateEdgeSelection
from cccpm.simulation.mediator_simulation import generate_confound_simulation
import matplotlib.pyplot as plt
import warnings
import torch
import time

warnings.filterwarnings('ignore')

import logging
import shutil, os
import gc


cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# cuda = torch.device('cpu')
cudastr = 'gpu'

# ── Timing state ──────────────────────────────────────────────────────────────
_timing = {
    'run_start': None,       # time.perf_counter() at script start
    'condition_times': [],   # elapsed seconds per condition (triplet)
}

def _ts():
    """Seconds since script start."""
    return time.perf_counter() - _timing['run_start']


def _fmt(seconds):
    """Format seconds as mm:ss.s"""
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:04.1f}"


def _print_timing_header():
    device_label = 'GPU (CUDA)' if torch.cuda.is_available() and cudastr == 'gpu' else 'CPU'
    print(f"\n{'=' * 60}")
    print(f"  DEVICE: {device_label}")
    print(f"  Script started. Timing begins now.")
    print(f"{'=' * 60}\n")


def _print_run_timing(label, elapsed):
    print(f"    ⏱  {label:<22} {elapsed:.2f}s  (total so far: {_fmt(_ts())})")


def _print_condition_summary(r2_target, rho, t_base, t_partial, t_cr):
    t_condition = t_base + t_partial + t_cr
    _timing['condition_times'].append(t_condition)
    n = len(_timing['condition_times'])
    avg = sum(_timing['condition_times']) / n
    print(f"  ── Condition summary  R2={r2_target}  ρ={rho} ──")
    print(f"     baseline={t_base:.2f}s  partial={t_partial:.2f}s  CR={t_cr:.2f}s  │  condition total={t_condition:.2f}s")
    print(f"     cumulative total: {_fmt(_ts())}  │  avg per condition: {avg:.2f}s  │  conditions done: {n}")


def _print_final_summary():
    total = _ts()
    n = len(_timing['condition_times'])
    avg = sum(_timing['condition_times']) / n if n else 0
    slowest = max(_timing['condition_times']) if n else 0
    fastest = min(_timing['condition_times']) if n else 0
    print(f"\n{'=' * 60}")
    print(f"  TIMING SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total wall time   : {_fmt(total)} ({total:.1f}s)")
    print(f"  Conditions run    : {n}")
    print(f"  Avg per condition : {avg:.2f}s")
    print(f"  Fastest condition : {fastest:.2f}s")
    print(f"  Slowest condition : {slowest:.2f}s")
    print(f"{'=' * 60}\n")


# ── Unchanged helpers ─────────────────────────────────────────────────────────
def cleanup_results():
    if os.path.exists('./results'):
        shutil.rmtree('./results')
    os.makedirs('./results', exist_ok=True)


def _silence_cccpm():
    logging.getLogger('cccpm').setLevel(logging.CRITICAL)
    logging.getLogger().setLevel(logging.CRITICAL)


def _restore_logging():
    logging.getLogger().setLevel(logging.WARNING)


def get_r2(cpm_obj):
    print(cpm_obj.results_manager.agg_results.index)
    val = cpm_obj.results_manager.agg_results.loc[
        'connectome', 'both']['explained_variance_score']['mean']
    return float(val)


def run_baseline(X, y, c, cv):
    ue = UnivariateEdgeSelection(
        edge_statistic='pearson',
        edge_selection=[PThreshold(threshold=0.05, correction=[None])],
        t_test_filter=False)
    cpm = CPMAnalysis(results_directory='./results',
                      cv=cv, edge_selection=ue,
                      n_permutations=1, select_stable_edges=False, device=cudastr)
    _silence_cccpm()
    X = torch.tensor(X).to(torch.device(cuda))
    y = torch.tensor(y).to(torch.device(cuda))
    c = torch.tensor(c).to(torch.device(cuda))
    t0 = time.perf_counter()
    cpm.run(X=X, y=y, covariates=c)
    elapsed = time.perf_counter() - t0
    _restore_logging()
    r2 = get_r2(cpm)
    del cpm
    gc.collect()
    _print_run_timing('baseline', elapsed)
    return r2, elapsed


def run_partial(X, y, c, cv):
    ue = UnivariateEdgeSelection(
        edge_statistic='pearson_partial',
        edge_selection=[PThreshold(threshold=0.05, correction=[None])],
        t_test_filter=False)
    cpm = CPMAnalysis(results_directory='./results',
                      cv=cv, edge_selection=ue,
                      n_permutations=1, select_stable_edges=False, device=cudastr)
    _silence_cccpm()
    X = torch.tensor(X).to(torch.device(cuda))
    y = torch.tensor(y).to(torch.device(cuda))
    c = torch.tensor(c).to(torch.device(cuda))
    t0 = time.perf_counter()
    cpm.run(X=X, y=y, covariates=c)
    elapsed = time.perf_counter() - t0
    _restore_logging()
    r2 = get_r2(cpm)
    _print_run_timing('partial', elapsed)
    return r2, elapsed


def run_cr(X, y, c, cv):
    ue = UnivariateEdgeSelection(
        edge_statistic='pearson',
        edge_selection=[PThreshold(threshold=0.05, correction=[None])],
        t_test_filter=False)
    cpm = CPMAnalysis(results_directory='./results',
                      cv=cv, edge_selection=ue,
                      n_permutations=1, calculate_residuals=True,
                      select_stable_edges=False, device=cudastr)
    _silence_cccpm()
    X = torch.tensor(X).to(torch.device(cuda))
    y = torch.tensor(y).to(torch.device(cuda))
    c = torch.tensor(c).to(torch.device(cuda))
    t0 = time.perf_counter()
    cpm.run(X=X, y=y, covariates=c)
    elapsed = time.perf_counter() - t0
    _restore_logging()
    r2 = get_r2(cpm)
    _print_run_timing('CR', elapsed)
    return r2, elapsed


import psutil, os


def log_mem(label=""):
    mb = psutil.Process(os.getpid()).memory_info().rss / 1e6
    print(f"[MEM {label}] {mb:.0f} MB")


# ── Grid ──────────────────────────────────────────────────────────────────────
R2_TARGETS = [0.09, 0.36, 0.81]
RHO_CONFOUNDS = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
N_SIMULATIONS = 10

_timing['run_start'] = time.perf_counter()
_print_timing_header()

records = []

for sim_idx in range(N_SIMULATIONS):
    cleanup_results()
    sim_seed = sim_idx * 100
    cv_seed = sim_idx
    CV = RepeatedKFold(n_splits=2, n_repeats=1, random_state=cv_seed)

    print(f"\n{'#' * 60}")
    print(f"  SIMULATION {sim_idx + 1}/{N_SIMULATIONS}  (data seed={sim_seed})  │  elapsed: {_fmt(_ts())}")
    print(f"{'#' * 60}")

    for r2_target in R2_TARGETS:
        for rho in RHO_CONFOUNDS:
            print(f"  R2_TARGET={r2_target}  RHO_CONFOUND={rho}", end="  ", flush=True)

            X, y, c = generate_confound_simulation(
                n_samples=1000, n_features=105,
                sparsity=0.8, x_collinearity=0.7,
                target_r2=r2_target, confounder_rho=rho,
                random_state=sim_seed)

            print()  # newline before per-run timing lines
            r2_base,    t_base    = run_baseline(X, y, c, CV)
            r2_partial, t_partial = run_partial(X, y, c, CV)
            r2_cr,      t_cr      = run_cr(X, y, c, CV)

            if rho == 1.0:
                r2_theory = 0.0
            else:
                r2_theory = (r2_target * (1 - rho ** 2) ** 2) / (1 - (rho ** 2 * r2_target))

            records.append(dict(
                simulation=sim_idx,
                r2_target=r2_target,
                rho=rho,
                r2_cpm_raw=r2_base,
                r2_partial=r2_partial,
                r2_cr=r2_cr,
                ExpectedAfterRemoval=r2_theory,
            ))
            print(f"     raw={r2_base:.4f}  partial={r2_partial:.4f}  CR={r2_cr:.4f}")
            _print_condition_summary(r2_target, rho, t_base, t_partial, t_cr)

            log_mem(f"sim={sim_idx} r2={r2_target} rho={rho}")

_print_final_summary()

df = pd.DataFrame(records)
df.to_csv('./results/simulation_results.csv', index=False)
print("\nAll simulations done. Results saved to ./results/simulation_results.csv")
print(df.to_string())

# ── Aggregate: mean & SD over simulations ────────────────────────────────────
agg = (df
       .groupby(['r2_target', 'rho'])[['r2_cpm_raw', 'r2_partial', 'r2_cr', 'ExpectedAfterRemoval']]
       .agg(['mean', 'std'])
       .reset_index())

agg.columns = ['_'.join(c).strip('_') for c in agg.columns]

# ── Plotting ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    len(R2_TARGETS), 1,
    figsize=(14, 14),
    sharex=True
)

bar_width = 0.22
colors = ['#4C72B0', '#DD8452', '#55A868']

for idx, r2 in enumerate(R2_TARGETS):
    ax = axes[idx]
    sub = agg[agg['r2_target'] == r2].reset_index(drop=True)
    x = np.arange(len(RHO_CONFOUNDS))

    ax.bar(
        x - bar_width, sub['r2_cpm_raw_mean'],
        width=bar_width, color=colors[0], label='Raw CPM',
        yerr=sub['r2_cpm_raw_std'], capsize=4,
        error_kw=dict(elinewidth=1.2, ecolor='black', alpha=0.7)
    )
    ax.bar(
        x, sub['r2_partial_mean'],
        width=bar_width, color=colors[1], label='Partial Corr',
        yerr=sub['r2_partial_std'], capsize=4,
        error_kw=dict(elinewidth=1.2, ecolor='black', alpha=0.7)
    )
    ax.bar(
        x + bar_width, sub['r2_cr_mean'],
        width=bar_width, color=colors[2], label='Confound Regression',
        yerr=sub['r2_cr_std'], capsize=4,
        error_kw=dict(elinewidth=1.2, ecolor='black', alpha=0.7)
    )

    ax.plot(
        x, [r2] * len(x),
        'ko-', linewidth=2,
        label='True signal' if idx == 0 else None
    )
    ax.plot(
        x, sub['ExpectedAfterRemoval_mean'],
        'r--', linewidth=2,
        label='Expected after deconfounding' if idx == 0 else None
    )

    ax.set_title(f'True signal R² = {r2}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prediction R²')
    ax.set_xticks(x)
    ax.set_xticklabels(RHO_CONFOUNDS)
    ax.grid(alpha=0.3, axis='y')

axes[-1].set_xlabel('Confound strength (ρ)', fontsize=12)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=10)
fig.suptitle(
    f'CPM confound simulation — mean ± SD over {N_SIMULATIONS} runs',
    fontsize=13, y=0.98
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('./results/simulation_bar_chart.pdf', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved to ./results/simulation_bar_chart.pdf")