import pytest
import subprocess
import sys
import os
from pathlib import Path

# 1. Locate the examples folder relative to this test file
#    (Assumes structure: root/tests/test_examples.py -> root/examples/)
REPO_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
SRC_DIR = REPO_ROOT / "src"  # <--- Define the src path

# 2. Define the examples to be tested
EXAMPLE_SCRIPTS = [
    "example_simulated_data.py"
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