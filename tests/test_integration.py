import pytest
import subprocess
import sys
import os
from pathlib import Path

# 1. Locate the examples folder relative to this test file
#    (Assumes structure: root/tests/test_examples.py -> root/examples/)
REPO_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"  # <--- Define the src path

# 2. Define the examples to be tested
EXAMPLE_SCRIPTS = [
    "regression_quickstart.py",
    "classification_quickstart.py",
]


@pytest.mark.parametrize("script_name", EXAMPLE_SCRIPTS)
def test_example_script_runs(script_name):
    script_path = EXAMPLES_DIR / script_name

    # 1. Prepare the environment variables
    env = os.environ.copy()

    # 2. Add the 'src' directory to PYTHONPATH
    # This ensures 'import cccpm' works inside the script
    env["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    # 3. Set Matplotlib to non-interactive mode
    env["MPLBACKEND"] = "Agg"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=EXAMPLES_DIR,
        capture_output=True,
        text=True,
        env=env  # <--- Pass the modified env here
    )

    assert result.returncode == 0, f"Script crashed:\n{result.stderr}"

# ---------------------------------------------------------------------------
# Not an example -- `scripts/confound_inflation_demo.py` is the package's own
# demonstration that partial-correlation selection does not fully deconfound, on
# data with an analytically known answer. It lives in scripts/ rather than
# examples/ because a 340-line parameter sweep is evidence, not a template.
#
# Too slow for CI at full size, so it is exercised on a minimal config.
#
# Importing them is not enough: the breakages these catch -- a tqdm stand-in
# that no longer matches how the toolbox constructs it, and a scalar extraction
# that silently became a Series -- only fire once the sweep actually runs. Both
# had been broken for a while precisely because nothing called them.
# ---------------------------------------------------------------------------

def test_confound_inflation_demo_sweep_runs():
    import importlib
    import logging
    import sys

    # The demo silences logging at module level (it runs a large sweep). That is
    # a process-global switch, so importing it here would leave every later test
    # without a cpm_log.txt -- which is where the report reads its configuration
    # block from. Restore it afterwards.
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
