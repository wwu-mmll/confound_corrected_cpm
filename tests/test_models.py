"""
Tests for all CPM model classes: LinearCPM, DecisionTreeCPM, RandomForestCPM, GAMCPM.
Validates the tensor-based pipeline interface (fit/predict shapes, chaining,
get_network_strengths) and full pipeline integration for non-linear models.
"""

import numpy as np
import pytest
import torch
from sklearn.model_selection import KFold

from cccpm.constants import Models, Networks, TaskType
from cccpm.models.linear_model import LinearCPM
from cccpm.models.nonlinear_models import DecisionTreeCPM, RandomForestCPM, GAMCPM
from cccpm.cpm_analysis import CPMAnalysis
from cccpm.edge_selection import UnivariateEdgeSelection, PThreshold


ALL_MODELS = [LinearCPM, DecisionTreeCPM, RandomForestCPM, GAMCPM]
NONLINEAR_MODELS = [DecisionTreeCPM, RandomForestCPM, GAMCPM]


@pytest.fixture
def simple_data():
    """Small synthetic dataset for unit tests."""
    rng = np.random.RandomState(42)
    N_samples, N_features, N_runs = 30, 10, 3
    X = rng.randn(N_samples, N_features).astype(np.float32)
    y = (X[:, :3].sum(axis=1, keepdims=True) + rng.randn(N_samples, 1) * 0.1).astype(np.float32)
    y = np.tile(y, (1, N_runs))
    covariates = rng.randn(N_samples, 2).astype(np.float32)
    edges = torch.zeros(N_features, 2, N_runs, dtype=torch.bool)
    edges[:5, Networks.positive, :] = True
    edges[5:, Networks.negative, :] = True
    return X, y, covariates, edges


# ============================================================
# TestModelInterface — parametrized over all 4 model classes
# ============================================================

class TestModelInterface:

    @pytest.mark.parametrize("model_cls", ALL_MODELS)
    def test_fit_predict_shape(self, model_cls, simple_data):
        """predict returns [N_samples, N_models, N_networks, N_runs]."""
        X, y, cov, edges = simple_data
        N_samples, N_runs = y.shape

        model = model_cls(edges=edges, device='cpu', task_type=TaskType.regression)
        preds = model.fit(X, y, cov).predict(X, cov)

        assert preds.shape == (N_samples, len(Models), len(Networks), N_runs)

    @pytest.mark.parametrize("model_cls", ALL_MODELS)
    def test_single_run(self, model_cls, simple_data):
        """With a 1-column y the last dimension should be 1."""
        X, y, cov, edges = simple_data
        N_samples = X.shape[0]

        y_single = y[:, :1]
        edges_single = edges[:, :, :1]

        model = model_cls(edges=edges_single, device='cpu', task_type=TaskType.regression)
        preds = model.fit(X, y_single, cov).predict(X, cov)

        assert preds.shape == (N_samples, len(Models), len(Networks), 1)

    @pytest.mark.parametrize("model_cls", ALL_MODELS)
    def test_get_network_strengths(self, model_cls, simple_data):
        """get_network_strengths returns correct keys and tensor values."""
        X, y, cov, edges = simple_data

        model = model_cls(edges=edges, device='cpu', task_type=TaskType.regression)
        model.fit(X, y, cov)
        ns = model.get_network_strengths(X, cov)

        assert set(ns.keys()) == {"connectome", "residuals"}
        for group in ["connectome", "residuals"]:
            assert "positive" in ns[group]
            assert "negative" in ns[group]
            assert isinstance(ns[group]["positive"], torch.Tensor)
            assert isinstance(ns[group]["negative"], torch.Tensor)
            assert ns[group]["positive"].shape[0] == X.shape[0]

    @pytest.mark.parametrize("model_cls", ALL_MODELS)
    def test_chaining(self, model_cls, simple_data):
        """model.fit(...).predict(...) works (fit returns self)."""
        X, y, cov, edges = simple_data
        model = model_cls(edges=edges, device='cpu', task_type=TaskType.regression)
        result = model.fit(X, y, cov).predict(X, cov)
        assert isinstance(result, torch.Tensor)


# ============================================================
# TestNonLinearModelsWithPipeline
# ============================================================

class TestNonLinearModelsWithPipeline:

    @pytest.mark.parametrize("model_cls", NONLINEAR_MODELS)
    def test_full_pipeline_run(self, model_cls, tmp_path):
        """CPMAnalysis with each non-linear model runs end-to-end on simulated data."""
        from cccpm.simulation.simulate_simple import simulate_confounded_data_chyzhyk

        X, y, covariates = simulate_confounded_data_chyzhyk(n_samples=60, n_features=45)

        edge_selection = UnivariateEdgeSelection(
            edge_statistic='pearson',
            edge_selection=[PThreshold(threshold=[0.05], correction=[None])],
        )

        cpm = CPMAnalysis(
            results_directory=str(tmp_path),
            task_type='regression',
            cpm_model=model_cls,
            cv=KFold(n_splits=3, shuffle=True, random_state=42),
            edge_selection=edge_selection,
            n_permutations=0,
            impute_missing_values=True,
        )

        cpm.run(X, y, covariates)

        assert cpm.results_manager is not None
        assert cpm.results_manager.agg_results is not None
