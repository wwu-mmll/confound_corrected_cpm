"""Guards the dependency direction from a past split: the numeric core used to
import `cccpm.reporting.plots.plots`, dragging matplotlib/seaborn into every
module that touched the maths. This keeps it from regressing."""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "cccpm"

# The numeric core: everything except the reporting layer itself. `cpm_analysis`
# is the orchestrator and is allowed to build reports, so it is exempt.
CORE_MODULES = sorted(
    p for p in SRC.rglob("*.py")
    if "reporting" not in p.relative_to(SRC).parts and p.name != "cpm_analysis.py"
)


def _imported_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.module


def _offenders(predicate):
    """Core modules whose imports match ``predicate``."""
    # Guards against CORE_MODULES silently ending up empty (e.g. a path typo).
    assert CORE_MODULES, f"no core modules found under {SRC}"

    found = {}
    for module in CORE_MODULES:
        hits = [m for m in _imported_modules(module) if predicate(m)]
        if hits:
            found[module.name] = hits
    return found


def test_numeric_core_does_not_import_reporting():
    offenders = _offenders(
        lambda m: m == "cccpm.reporting" or m.startswith("cccpm.reporting."))
    assert not offenders, (
        f"these modules import the reporting layer: {offenders}. Move the "
        f"plotting code into cccpm/reporting/ instead -- see "
        f"cccpm/reporting/data_insights.py for the pattern."
    )


def test_numeric_core_does_not_import_plotting_stack():
    stack = {"matplotlib", "seaborn", "netplotbrain", "pycirclize", "plotly"}
    offenders = _offenders(lambda m: m.split(".")[0] in stack)
    assert not offenders, f"these modules import the plotting stack: {offenders}."
