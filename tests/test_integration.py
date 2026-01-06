import pytest
import subprocess
import sys
import os
from pathlib import Path

# 1. Locate the examples folder relative to this test file
#    (Assumes structure: root/tests/test_examples.py -> root/examples/)
REPO_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"

# 2. Define the examples to be tested
EXAMPLE_SCRIPTS = [
    "example_simulated_data.py"
]


@pytest.mark.parametrize("script_name", EXAMPLE_SCRIPTS)
def test_example_script_runs(script_name):
    script_path = EXAMPLES_DIR / script_name

    # Ensure the file actually exists before running
    assert script_path.exists(), f"Example script not found: {script_name}"

    # 3. Run the script as a subprocess
    #    sys.executable ensures we use the SAME python interpreter that runs the tests
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=EXAMPLES_DIR,  # Run it from the examples folder (standard behavior)
        capture_output=True,  # Capture stdout/stderr so we can see errors
        text=True,  # Decode output as string
        env={**os.environ, "MPLBACKEND": "Agg"}  # <--- CRITICAL (See below)
    )

    # 4. Assert success
    #    If returncode is not 0, the script crashed.
    assert result.returncode == 0, f"Script crashed:\n{result.stderr}"