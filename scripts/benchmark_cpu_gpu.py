"""
CPU-vs-GPU timing harness for the CCCPM hot path.

Times the two numerically heavy steps — univariate edge selection
(`EdgeStatistic.fit_transform`) and linear-model fit+predict (`LinearCPM`) — on
CPU and, when available, CUDA, across problem sizes and permutation counts.

Why this exists: users observe that CPU and GPU runs take almost the same wall
time. This harness shows why. The GPU only wins when there is enough parallel
work to hide kernel-launch and host↔device-transfer overhead — i.e. large
permutation batches and/or many edges. At `perms=1` (a plain run with no
permutation test) the matrices are small and the GPU is typically no faster (or
slower) than the CPU. Sweep `--perms` to see the crossover.

Note: this measures the compute kernels only. A full `CPMAnalysis.run` also
spends fixed, CPU-bound time on data validation, imputation, permutation
generation, the connected-component filter (networkx), scoring, CSV/IO and the
HTML report — none of which the GPU accelerates — so end-to-end speedups are
smaller than the kernel speedups reported here.

Usage:
    python scripts/benchmark_cpu_gpu.py
    python scripts/benchmark_cpu_gpu.py --nodes 100 200 --samples 300 --perms 1 100 1000
    python scripts/benchmark_cpu_gpu.py --repeat 5
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Ensure the src layout is importable when running from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from cccpm.edge_selection import EdgeStatistic
from cccpm.models.linear_model import LinearCPM


def _make_data(n_samples, n_nodes, n_perms, seed=0):
    """Synthetic connectome-shaped data and a random edge mask for the model."""
    rng = np.random.RandomState(seed)
    n_features = n_nodes * (n_nodes - 1) // 2
    X = rng.randn(n_samples, n_features).astype(np.float32)
    Y = rng.randn(n_samples, n_perms).astype(np.float32)
    cov = rng.randn(n_samples, 3).astype(np.float32)

    # ~5% of edges selected per network (positive / negative), same across perms.
    edges = torch.zeros(n_features, 2, n_perms, dtype=torch.bool)
    k = max(1, n_features // 20)
    for layer in (0, 1):
        idx = rng.choice(n_features, size=k, replace=False)
        edges[idx, layer, :] = True
    return X, Y, cov, edges


def _time(fn, device, n_repeat):
    """Mean seconds per call, with a warmup and CUDA synchronisation."""
    fn()  # warmup (kernel compile / allocator / transfer paths)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_repeat):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_repeat


def run_benchmark(nodes, samples, perms, devices, n_repeat=3):
    """Return a list of timing rows (dicts) for the size × device grid."""
    rows = []
    for n_nodes in nodes:
        for n_samples in samples:
            for n_perms in perms:
                X, Y, cov, edges = _make_data(n_samples, n_nodes, n_perms)
                for device in devices:
                    dev = torch.device(device)
                    stat = EdgeStatistic(selection_statistic="pearson")
                    edge_fn = lambda: stat.fit_transform(
                        X=X, y=Y, covariates=cov, device=dev)
                    edges_dev = edges.to(dev)
                    model = LinearCPM(edges=edges_dev, device=device)
                    model_fn = lambda: model.fit(X, Y, cov).predict(X, cov)

                    rows.append({
                        "nodes": n_nodes,
                        "features": n_nodes * (n_nodes - 1) // 2,
                        "samples": n_samples,
                        "perms": n_perms,
                        "device": device,
                        "edge_ms": _time(edge_fn, dev, n_repeat) * 1e3,
                        "model_ms": _time(model_fn, dev, n_repeat) * 1e3,
                    })
    return rows


def _print_table(rows):
    header = f"{'nodes':>5} {'feat':>7} {'N':>5} {'perms':>6} {'device':>6} " \
             f"{'edge ms':>9} {'model ms':>9} {'total ms':>9} {'vs CPU':>7}"
    print(header)
    print("-" * len(header))
    # group by (nodes, samples, perms) so we can print the GPU speedup vs CPU.
    cpu_total = {}
    for r in rows:
        if r["device"] == "cpu":
            cpu_total[(r["nodes"], r["samples"], r["perms"])] = r["edge_ms"] + r["model_ms"]
    for r in rows:
        total = r["edge_ms"] + r["model_ms"]
        base = cpu_total.get((r["nodes"], r["samples"], r["perms"]))
        speedup = f"{base / total:5.2f}x" if (base and r["device"] != "cpu") else ""
        print(f"{r['nodes']:>5} {r['features']:>7} {r['samples']:>5} {r['perms']:>6} "
              f"{r['device']:>6} {r['edge_ms']:>9.2f} {r['model_ms']:>9.2f} "
              f"{total:>9.2f} {speedup:>7}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nodes", type=int, nargs="+", default=[100, 200])
    parser.add_argument("--samples", type=int, nargs="+", default=[300])
    parser.add_argument("--perms", type=int, nargs="+", default=[1, 100, 1000])
    parser.add_argument("--repeat", type=int, default=3, help="timed repeats per cell")
    args = parser.parse_args()

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}\n")
    else:
        print("CUDA not available — timing CPU only.\n")

    rows = run_benchmark(args.nodes, args.samples, args.perms, devices, args.repeat)
    _print_table(rows)

    if "cuda" in devices:
        print("\nRead-out: the GPU only beats the CPU once there is enough parallel "
              "work (large `perms` and/or `features`) to amortise launch + transfer "
              "overhead. Near perms=1 the two are expected to be comparable.")


if __name__ == "__main__":
    main()
