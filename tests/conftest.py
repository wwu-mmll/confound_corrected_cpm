# Non-interactive backend before anything imports pyplot: avoids _tkinter.TclError
# on CI runners with no Tcl/Tk (e.g. Windows). Must run before the cccpm imports below.
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
    return simulate_confounded_data_chyzhyk(n_samples=100, n_features=45)


@pytest.fixture(scope="function")
def simulated_classification_data():
    """Like simulated_data, with y binarized to 0/1."""
    X, y, covariates = simulate_confounded_data_chyzhyk(n_samples=100, n_features=45)
    y_binary = (y > np.median(y)).astype(float)
    return X, y_binary, covariates


@pytest.fixture(scope="function")
def cpm_instance(tmp_path):
    univariate_edge_selection = UnivariateEdgeSelection(
        selection_statistic='pearson',
        edge_selection=[PThreshold(threshold=[0.01, 0.05], correction=[None])]
    )

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


def report_text(results_directory):
    """The report as a reader sees it: markup and embedded base64 figures stripped,
    read as UTF-8 (not the Windows locale codec). Regression: stripping only
    `data:image/...;base64,` immediately followed by base64 missed matplotlib's
    SVG output (newline after the comma), letting stray "nan" bytes in the base64
    payload red the macOS/Windows CI jobs. Callers should still anchor a word
    search, e.g. re.search(r"\\bnan\\b", text, re.I)."""
    html = Path(results_directory, 'report.html').read_text(encoding='utf-8')
    without_code = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", html)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", without_code))
