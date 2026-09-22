# Force a non-interactive matplotlib backend before anything imports pyplot.
# The library only ever saves plots to disk, so tests never need a GUI backend;
# this avoids _tkinter.TclError on CI runners with a broken/absent Tcl/Tk
# (e.g. Windows hosted runners). Must run before the cccpm imports below.
import matplotlib
matplotlib.use("Agg")

import re
from pathlib import Path

import pytest
import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold
from cccpm.simulation.simulate_simple import simulate_confounded_data_chyzhyk
from cccpm.edge_selection import UnivariateEdgeSelection, PThreshold
from cccpm.cpm_analysis import CPMAnalysis


@pytest.fixture(scope="function")
def simulated_data():
    """
    Returns a standard tuple of (X, y, covariates) for testing.
    n_samples=100, n_features=45.
    """
    return simulate_confounded_data_chyzhyk(n_samples=100, n_features=45)


@pytest.fixture(scope="function")
def simulated_classification_data():
    """
    Returns a standard tuple of (X, y, covariates) for classification testing.
    n_samples=100, n_features=45. y is binarized to 0/1.
    """
    X, y, covariates = simulate_confounded_data_chyzhyk(n_samples=100, n_features=45)
    y_binary = (y > np.median(y)).astype(float)
    return X, y_binary, covariates


@pytest.fixture(scope="function")
def cpm_instance(tmp_path):
    """
    Returns an initialized CPMAnalysis instance configured with a
    temporary results directory.
    """
    univariate_edge_selection = UnivariateEdgeSelection(
        selection_statistic='pearson',
        edge_selection=[PThreshold(threshold=[0.01, 0.05], correction=[None])]
    )

    # tmp_path is automatically provided by pytest and cleaned up afterwards
    return CPMAnalysis(
        results_directory=tmp_path,
        cv=KFold(n_splits=10, shuffle=True, random_state=42),
        inner_cv=ShuffleSplit(n_splits=1, random_state=42),
        edge_selection=univariate_edge_selection,
        n_permutations=2,
        impute_missing_values=True
    )


@pytest.fixture(scope="function")
def cpm_classification_instance(tmp_path):
    """
    Returns an initialized CPMAnalysis instance configured for classification.
    """
    univariate_edge_selection = UnivariateEdgeSelection(
        selection_statistic='pearson',
        edge_selection=[PThreshold(threshold=[0.05], correction=[None])]
    )

    return CPMAnalysis(
        results_directory=tmp_path,
        task_type='classification',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        edge_selection=univariate_edge_selection,
        n_permutations=0,
        impute_missing_values=True
    )


# ---------------------------------------------------------------------------
# Reading the HTML report in tests
# ---------------------------------------------------------------------------

def report_text(results_directory):
    """The report as a reader sees it: markup and embedded figures removed.

    Two traps this closes, both of which have bitten.

    *Base64 is not prose.* A report inlines ~100-180 KB of base64 per figure.
    Whether the three letters of "nan" happen to occur somewhere in those bytes
    is chance -- measured in 2 of 9 real reports -- and the bytes differ per
    operating system because the rendered figures do. Stripping only
    ``data:image/...;base64,`` followed *immediately* by base64 misses
    matplotlib's SVG output, which writes a newline after the comma; that near
    miss turned the macOS and Windows CI jobs red while Linux passed. Dropping
    all markup removes the payloads wholesale, because they live in attributes.

    *The locale codec.* ``open(path).read()`` decodes as cp1252 on Windows,
    while the report is UTF-8.

    Callers looking for a word should still anchor it --
    ``re.search(r"\\bnan\\b", text, re.I)`` -- so that no accidental substring
    can revive the same class of failure.
    """
    html = Path(results_directory, 'report.html').read_text(encoding='utf-8')
    without_code = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", html)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", without_code))
