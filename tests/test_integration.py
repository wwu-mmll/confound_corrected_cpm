"""End-to-end checks: the example scripts and the confound-inflation demo sweep."""

import pytest
import subprocess
import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"

EXAMPLE_SCRIPTS = [
    "regression_quickstart.py",
    "classification_quickstart.py",
]


@pytest.mark.parametrize("script_name", EXAMPLE_SCRIPTS)
def test_example_script_runs(script_name):
    script_path = EXAMPLES_DIR / script_name

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=EXAMPLES_DIR,
        capture_output=True,
        text=True,
        env=env
    )

    assert result.returncode == 0, f"Script crashed:\n{result.stderr}"


# scripts/confound_inflation_demo.py is evidence (a 340-line sweep), not a
# template, hence scripts/ not examples/. Run on a minimal config: full size
# is too slow for CI, and importing alone wouldn't catch bugs that only fire
# once the sweep actually runs (e.g. a scalar silently becoming a Series).
def test_confound_inflation_demo_sweep_runs():
    import importlib
    import logging
    import sys

    # Demo disables logging globally for its sweep; later tests need it back
    # (the report reads its configuration block from cpm_log.txt).
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        demo = importlib.import_module("confound_inflation_demo")
        demo.R2_TARGETS = (0.36,)
        demo.KAPPAS = (0.6,)
        demo.N_SIMS = 1
        demo.N_SAMPLES = 400
        df, edges = demo.run_sweep()
    finally:
        sys.path.remove(str(SCRIPTS_DIR))
        logging.disable(logging.NOTSET)

    assert len(df) == 1
    row = df.iloc[0]
    # The point of the demo: the naive model is inflated above the true value,
    # and controlling the connectome recovers it.
    assert row["connectome_raw"] > row["true_r2"] + 0.1
    assert abs(row["connectome_residualizedX"] - row["true_r2"]) < 0.06

    # And naive selection keeps the confound-only edges that partial rejects.
    by_selection = edges.set_index("selection")["n_confound_only"]
    assert by_selection["raw"] > by_selection["partial"]
