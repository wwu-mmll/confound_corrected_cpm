"""Smoke test for the CPU-vs-GPU benchmark harness (scripts/benchmark_cpu_gpu.py).

Keeps the harness importable and runnable; it does not assert on timings (those
are hardware-dependent), only that the grid is produced with sane fields.
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_cpu_gpu as bench  # noqa: E402


def test_run_benchmark_cpu_smoke():
    rows = bench.run_benchmark(
        nodes=[10], samples=[40], perms=[1, 4], devices=["cpu"], n_repeat=1)

    assert len(rows) == 2  # 1 node-size × 1 sample-size × 2 perm counts
    for r in rows:
        assert r["device"] == "cpu"
        assert r["features"] == 10 * 9 // 2
        assert r["edge_ms"] > 0
        assert r["model_ms"] > 0
