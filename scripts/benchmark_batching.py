"""
Does batching over CV FOLDS earn its complexity?

Fold batching is the only reason the codebase needs zero-padding, per-fold
validity masks, `build_fold_batch`, and most of `batch_planning.py` -- that
machinery leaks into the signature of every solver in the package. Batching
over permutations needs none of it, because permutations add a column to `y`
rather than a ragged sample axis.

So the question this answers is narrow: how much wall time does the folds axis
actually buy, on top of what permutation batching already gives?

The cost model says folds are expensive per batch slot. At Shen-atlas size one
extra fold needs its own copy of X_train/X_test (~70 MB) while one extra
permutation reuses X (~300 KB) -- roughly 200x cheaper. Since the planner fills
params -> folds -> perms in that order, it may be spending the memory budget on
the expensive axis first.

Three configurations, same workload:

  1. current    -- planner as shipped (params -> folds -> perms)
  2. perms-only -- plan.folds forced to 1 (what removing fold batching gives)
  3. perms-first-- planner reordered to perms -> params -> folds

What is timed is `CPMAnalysis._single_run`, i.e. the cross-validation hot path
that batching governs: edge selection, model fit/predict, scoring, storage. The
HTML report, permutation inference and data-insight plots are excluded on
purpose -- they are fixed CPU-bound costs that dilute the comparison and are
unaffected by any of this.

Decision rule (from the refactoring plan): keep fold batching only if config 1
or 3 beats config 2 by MORE THAN 2x. Anything less does not justify the
padding/masking machinery.

Usage:
    python scripts/benchmark_batching.py                  # realistic workload
    python scripts/benchmark_batching.py --quick          # small smoke run
    python scripts/benchmark_batching.py --device cpu
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from sklearn.model_selection import KFold

import cccpm.cpm_analysis as cpm_analysis_module
from cccpm.batch_planning import BatchPlan, make_cpm_cost_fn, available_memory_bytes
from cccpm.constants import TaskType
from cccpm.cpm_analysis import CPMAnalysis
from cccpm.edge_selection import PThreshold, UnivariateEdgeSelection
from cccpm.simulation.simulate_sem import simulate_data_given_kappa


def _plan_with_priority(order):
    """A `plan_batch_sizes` clone that grows dimensions in a chosen order.

    Same greedy binary search as the production planner; only the order in
    which the three axes are grown differs. Keeping it here rather than
    parameterising production code means the benchmark cannot change shipped
    behaviour.
    """
    def plan_batch_sizes(n_params, n_folds, n_perms, cost_fn, available_bytes,
                         safety_factor=0.8):
        budget = available_bytes * safety_factor
        totals = {'params': max(1, n_params), 'folds': max(1, n_folds),
                  'perms': max(1, n_perms)}
        sizes = {'params': 1, 'folds': 1, 'perms': 1}

        def fits(candidate):
            return cost_fn(candidate['params'], candidate['folds'],
                           candidate['perms']) <= budget

        for dim in order:
            lo, hi = 1, totals[dim]
            if hi <= 1:
                continue
            candidate = dict(sizes)
            candidate[dim] = hi
            if fits(candidate):
                sizes[dim] = hi
                continue
            best = 1
            while lo <= hi:
                mid = (lo + hi) // 2
                candidate[dim] = mid
                if fits(candidate):
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            sizes[dim] = best
        return BatchPlan(**sizes)
    return plan_batch_sizes


def _plan_folds_forced_to_one(base_planner):
    """Wrap a planner so the folds axis is never batched."""
    def plan_batch_sizes(*args, **kwargs):
        plan = base_planner(*args, **kwargs)
        return BatchPlan(params=plan.params, folds=1, perms=plan.perms)
    return plan_batch_sizes


CONFIGS = {
    "current": lambda: _plan_with_priority(("params", "folds", "perms")),
    "perms-only": lambda: _plan_folds_forced_to_one(
        _plan_with_priority(("params", "folds", "perms"))),
    "perms-first": lambda: _plan_with_priority(("perms", "params", "folds")),
}


def build_data(n_features, n_samples, seed=42):
    sim = simulate_data_given_kappa(
        R2_X_y=0.4, kappa=0.3,
        n_features=n_features,
        n_features_informative=max(10, n_features // 130),
        n_pure_signal_features=max(5, n_features // 260),
        n_confound_only_features=max(5, n_features // 260),
        n_confounds=2, n_samples=n_samples, random_state=seed,
    )
    # y comes back as (n, 1); ravel so the permutation stack below is 2D.
    return sim["X"], np.asarray(sim["y"]).ravel(), sim["Z"]


def run_one(name, X, y, covariates, n_perms, n_folds, device, results_dir):
    """Time _single_run under one planner configuration."""
    planner = CONFIGS[name]()
    original = cpm_analysis_module.plan_batch_sizes
    cpm_analysis_module.plan_batch_sizes = planner

    observed = {}
    try:
        cpm = CPMAnalysis(
            results_directory=str(results_dir / name),
            cv=KFold(n_splits=n_folds, shuffle=True, random_state=42),
            edge_selection=UnivariateEdgeSelection(
                selection_statistic='pearson',
                edge_selection=[PThreshold(threshold=[0.05], correction=[None])],
            ),
            inner_cv=None,          # forces the batched outer-fold path
            n_permutations=0,       # permutations supplied directly below
            device=device,
        )
        # run() normally auto-detects this; _single_run is called directly here.
        cpm.task_type = TaskType.regression

        # Permutation columns are what the perms axis batches over.
        rng = np.random.default_rng(0)
        y_perms = np.stack([rng.permutation(y) for _ in range(n_perms)], axis=1)

        # Report the plan this configuration actually chose.
        cost_fn = make_cpm_cost_fn(
            int(len(y) * (n_folds - 1) / n_folds), int(len(y) / n_folds),
            X.shape[1], covariates.shape[1])
        plan = planner(1, n_folds, n_perms, cost_fn,
                       available_memory_bytes(torch.device(device)))
        observed["plan"] = plan

        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        cpm._single_run(X=X, y=y_perms, covariates=covariates, perm_run=True)
        if device == "cuda":
            torch.cuda.synchronize()
        observed["seconds"] = time.perf_counter() - start
        observed["peak_mb"] = (torch.cuda.max_memory_allocated() / 1e6
                               if device == "cuda" else float("nan"))
    finally:
        cpm_analysis_module.plan_batch_sizes = original

    return observed


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nodes", type=int, default=127,
                   help="connectome nodes (127 -> 8001 edges, as in profile_run.py)")
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--perms", type=int, default=1000)
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   choices=["cpu", "cuda"])
    p.add_argument("--quick", action="store_true",
                   help="small workload for a smoke test, not for deciding anything")
    p.add_argument("--out", type=Path,
                   default=Path("/tmp/cccpm_benchmark_batching"))
    args = p.parse_args()

    if args.quick:
        args.nodes, args.samples, args.perms, args.folds = 50, 200, 50, 5

    n_features = args.nodes * (args.nodes - 1) // 2
    print(f"device={args.device}  nodes={args.nodes} ({n_features} edges)  "
          f"samples={args.samples}  perms={args.perms}  folds={args.folds}")
    print("timing CPMAnalysis._single_run (the CV hot path); "
          "report/inference/IO excluded\n")

    X, y, covariates = build_data(n_features, args.samples)
    args.out.mkdir(parents=True, exist_ok=True)

    # Warm-up: the first timed configuration would otherwise absorb CUDA context
    # creation and cuBLAS autotuning, making whichever config ran first look ~18%
    # slower than an identically-planned config that ran later. Discard one run.
    print("  (warm-up, discarded) ...", end="", flush=True)
    run_one("perms-only", X, y, covariates, args.perms, args.folds,
            args.device, args.out)
    print(" done\n")

    results = {}
    for name in ("current", "perms-only", "perms-first"):
        print(f"  {name} ...", end="", flush=True)
        results[name] = run_one(name, X, y, covariates, args.perms, args.folds,
                                args.device, args.out)
        r = results[name]
        print(f" {r['seconds']:8.2f}s  plan(params={r['plan'].params}, "
              f"folds={r['plan'].folds}, perms={r['plan'].perms})"
              + (f"  peak {r['peak_mb']:.0f} MB" if args.device == "cuda" else ""))

    baseline = results["perms-only"]["seconds"]
    mem_baseline = results["perms-only"]["peak_mb"]
    print(f"\n{'config':<14}{'seconds':>9}{'speedup':>10}{'peak MB':>10}{'vs perms-only':>15}")
    for name, r in results.items():
        mem = r["peak_mb"]
        ratio = f"{mem / mem_baseline:.1f}x" if mem == mem else "n/a"
        mem_s = f"{mem:.0f}" if mem == mem else "n/a"
        print(f"{name:<14}{r['seconds']:>9.2f}{baseline / r['seconds']:>9.2f}x"
              f"{mem_s:>10}{ratio:>15}")

    best_with_folds = min(results["current"]["seconds"],
                          results["perms-first"]["seconds"])
    speedup = baseline / best_with_folds
    print(f"\nBest fold-batched config is {speedup:.2f}x faster than perms-only.")
    if speedup > 2.0:
        print("=> ABOVE the 2x bar: fold batching earns its complexity. Keep it.")
    else:
        print("=> BELOW the 2x bar: fold batching does not justify the padding/"
              "masking machinery. Remove it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
