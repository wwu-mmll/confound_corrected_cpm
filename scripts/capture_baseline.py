"""
Capture a numerical baseline of the quickstart outputs, for regression-testing
a refactor that is supposed to change nothing.

Runs both quickstart examples on every requested device and archives the files
that encode the actual numbers -- not the HTML report, not the plots, which
carry timestamps and are not meaningfully diffable.

Usage:
    poetry run python scripts/capture_baseline.py --out baselines/pre-refactor
    poetry run python scripts/capture_baseline.py --out baselines/post-phase4
    poetry run python scripts/capture_baseline.py --compare \
        baselines/pre-refactor baselines/post-phase4

`--compare` is the point of the whole thing: cv_results_summary.csv must match
to the 4 decimals it is written with, and stability_edges.npy must be bit
identical. Anything else is a regression to explain, not to tolerate.
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
# Files that encode numbers. Not every run produces all of them (permutation
# outputs only exist when n_permutations > 0).
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


def compare(ref_dir: Path, new_dir: Path) -> int:
    """Diff two baselines. Returns the number of mismatches found."""
    mismatches = 0
    checked = 0
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
            a, b = np.load(ref_file), np.load(new_file)
            if a.shape != b.shape:
                print(f"SHAPE    {rel}: {a.shape} vs {b.shape}")
                mismatches += 1
            elif not np.array_equal(a, b, equal_nan=True):
                worst = np.nanmax(np.abs(a - b))
                print(f"DIFFER   {rel}: max|diff| = {worst:.3e}")
                mismatches += 1
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
            elif not np.allclose(a[num], b[num], atol=5e-5, equal_nan=True):
                worst = np.nanmax(np.abs(a[num].to_numpy() - b[num].to_numpy()))
                print(f"DIFFER   {rel}: max|diff| = {worst:.3e} (tol 5e-5)")
                mismatches += 1

    print(f"\n{checked} file(s) compared, {mismatches} mismatch(es).")
    return mismatches


def main() -> int:
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
