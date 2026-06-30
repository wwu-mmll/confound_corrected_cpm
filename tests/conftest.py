import pytest
import numpy as np
import pandas as pd
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
        edge_statistic='pearson',
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
        edge_statistic='pearson',
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
