"""
Structural guards on the package layout.

These do not test behaviour; they protect the dependency direction established
when `utils.py` was split up. The numeric core used to import
`cccpm.reporting.plots.plots`, which dragged matplotlib and seaborn into every
module that touched the maths. Splitting the file fixed it once -- this test
keeps it fixed.
"""
import ast
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[1] / "src" / "cccpm"

# The numeric core: everything except the reporting layer itself. `cpm_analysis`
# is the orchestrator and is allowed to build reports, so it is exempt.
CORE_MODULES = sorted(
    p for p in SRC.rglob("*.py")
    if "reporting" not in p.relative_to(SRC).parts and p.name != "cpm_analysis.py"
)


def _imported_modules(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.module


@pytest.mark.parametrize("module", CORE_MODULES, ids=lambda p: str(p.name))
def test_numeric_core_does_not_import_reporting(module):
    """The dependency arrow points core -> reporting, never the other way."""
    offenders = [m for m in _imported_modules(module)
                 if m == "cccpm.reporting" or m.startswith("cccpm.reporting.")]
    assert not offenders, (
        f"{module.name} imports the reporting layer ({offenders}). Move the "
        f"plotting code into cccpm/reporting/ instead -- see "
        f"cccpm/reporting/data_insights.py for the pattern."
    )


@pytest.mark.parametrize("module", CORE_MODULES, ids=lambda p: str(p.name))
def test_numeric_core_does_not_import_plotting_stack(module):
    """Nor may it import matplotlib/seaborn directly."""
    offenders = [m for m in _imported_modules(module)
                 if m.split(".")[0] in {"matplotlib", "seaborn", "netplotbrain",
                                        "pycirclize", "plotly"}]
    assert not offenders, f"{module.name} imports the plotting stack ({offenders})."
