"""
Capture a numerical baseline of the quickstart outputs, to regression-test a
refactor that is supposed to change nothing.

Archives the files that encode actual numbers, not the HTML report or plots
(timestamped, not diffable). `--compare` always prints the observed max
difference per file, pass or fail — an earlier silent-on-success version let a
3.3e-6 drift sit undetected under threshold for two commits. CSV tolerance is
1e-5 (float32 accumulation order shifts with tensor shape across a refactor);
.npy edge/stability arrays must still be bit-identical.

Usage:
    poetry run python scripts/capture_baseline.py --out baselines/pre-refactor
    poetry run python scripts/capture_baseline.py --compare \
        baselines/pre-refactor baselines/post-phase4
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ["regression_quickstart", "classification_quickstart"]
# Not every run produces all of these (permutation outputs need n_permutations > 0).
ARTIFACTS = [
    "cv_results_summary.csv",
    "cv_results_full.csv",
    "p_values.csv",
    "cv_predictions.csv",
    "stability_edges.npy",
    "edges.npy",
    "stability_edges_significance.npy",
]


def run_and_archive(example: str, device: str, out_dir: Path) -> None:
    """Run one quickstart example on `device` and archive its numeric artifacts."""
    script = REPO_ROOT / "examples" / f"{example}.py"
    results = REPO_ROOT / "results" / example

    if results.exists():
        shutil.rmtree(results)

    env_note = "" if device == "cuda" else ' (CUDA_VISIBLE_DEVICES="")'
    print(f"  running {example} on {device}{env_note} ...", flush=True)
    env = None
    if device == "cpu":
        import os
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}

    proc = subprocess.run([sys.executable, str(script)], cwd=REPO_ROOT,
                          env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"{example} failed on {device} (exit {proc.returncode})")

    dest = out_dir / device / example
    dest.mkdir(parents=True, exist_ok=True)
    archived = 0
    for name in ARTIFACTS:
        src = results / name
        if src.exists():
            shutil.copy2(src, dest / name)
            archived += 1
    print(f"    archived {archived} artifact(s) -> {dest}")


CSV_TOLERANCE = 1e-5


def compare(ref_dir: Path, new_dir: Path) -> int:
    """Diff two baselines; returns the number of mismatches found."""
    mismatches = 0
    checked = 0
    worst_overall = 0.0
    worst_name = None
    for ref_file in sorted(ref_dir.rglob("*")):
        if not ref_file.is_file():
            continue
        rel = ref_file.relative_to(ref_dir)
        new_file = new_dir / rel
        if not new_file.exists():
            print(f"MISSING  {rel}")
            mismatches += 1
            continue

        checked += 1
        if ref_file.suffix == ".npy":
            # Discrete selections, not accumulated sums -> must be bit identical.
            a, b = np.load(ref_file), np.load(new_file)
            if a.shape != b.shape:
                print(f"SHAPE    {rel}: {a.shape} vs {b.shape}")
                mismatches += 1
                continue
            worst = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
            if not np.array_equal(a, b, equal_nan=True):
                print(f"DIFFER   {rel}: max|diff| = {worst:.3e} (must be exact)")
                mismatches += 1
            else:
                print(f"ok       {rel}: exact")
        else:
            a = pd.read_csv(ref_file)
            b = pd.read_csv(new_file)
            if a.shape != b.shape:
                print(f"SHAPE    {rel}: {a.shape} vs {b.shape}")
                mismatches += 1
                continue
            num = a.select_dtypes("number").columns
            if not a.drop(columns=num).equals(b.drop(columns=num)):
                print(f"DIFFER   {rel}: non-numeric columns differ")
                mismatches += 1
                continue
            worst = (float(np.nanmax(np.abs(a[num].to_numpy() - b[num].to_numpy())))
                     if len(num) else 0.0)
            if worst > CSV_TOLERANCE:
                print(f"DIFFER   {rel}: max|diff| = {worst:.3e} (tol {CSV_TOLERANCE:.0e})")
                mismatches += 1
            else:
                print(f"ok       {rel}: max|diff| = {worst:.3e}")

        if worst > worst_overall:
            worst_overall, worst_name = worst, rel

    print(f"\n{checked} file(s) compared, {mismatches} mismatch(es).")
    if worst_name is not None:
        print(f"largest difference anywhere: {worst_overall:.3e} ({worst_name})")
    return mismatches


def main() -> int:
    """Capture a baseline, or compare two existing ones."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="directory to write the baseline into")
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"],
                        choices=["cpu", "cuda"])
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("REF", "NEW"),
                        help="compare two previously captured baselines")
    args = parser.parse_args()

    if args.compare:
        return 1 if compare(*args.compare) else 0

    if not args.out:
        parser.error("either --out or --compare is required")

    devices = list(args.devices)
    if "cuda" in devices:
        import torch
        if not torch.cuda.is_available():
            print("CUDA unavailable; capturing CPU baseline only.")
            devices = [d for d in devices if d != "cuda"]

    args.out.mkdir(parents=True, exist_ok=True)
    for device in devices:
        print(f"[{device}]")
        for example in EXAMPLES:
            run_and_archive(example, device, args.out)
    print(f"\nBaseline written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
