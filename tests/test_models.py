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


def test_linear_cpm_defaults_to_cpu(simple_data):
    """LinearCPM must default to CPU so it works on machines without CUDA."""
    _, _, _, edges = simple_data
    model = LinearCPM(edges=edges)
    assert torch.device(model.device) == torch.device("cpu")


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


# ============================================================
# Batched-over-params/folds LinearCPM.fit_batched/predict_batched
# ============================================================

def _uneven_splits(n_total, n_folds, seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_total)
    edges = sorted(rng.choice(range(5, n_total - 5), size=n_folds - 1, replace=False))
    boundaries = [0] + list(edges) + [n_total]
    splits = []
    for i in range(n_folds):
        test_idx = idx[boundaries[i]:boundaries[i + 1]]
        train_idx = np.setdiff1d(idx, test_idx)
        splits.append((train_idx, test_idx))
    return splits


@pytest.mark.parametrize("task_type", [TaskType.regression, TaskType.classification])
def test_linear_cpm_batched_matches_looped_reference(task_type):
    """fit_batched/predict_batched, with multiple params AND padded (uneven)
    folds, must reproduce calling fit()/predict() once per (param, fold)
    pair on that fold's real (unpadded) rows and that param's edge mask."""
    from cccpm.utils import build_fold_batch

    torch.manual_seed(11)
    N_total, F, P_perms, C = 45, 8, 2, 2
    X_full = torch.randn(N_total, F, dtype=torch.float32)
    cov_full = torch.randn(N_total, C, dtype=torch.float32)
    if task_type == TaskType.classification:
        y_full = (torch.rand(N_total, P_perms) > 0.5).float()
    else:
        y_full = torch.randn(N_total, P_perms, dtype=torch.float32)

    splits = _uneven_splits(N_total, n_folds=3, seed=7)
    B_folds = len(splits)
    B_params = 2

    edges_per_param = []
    for pidx in range(B_params):
        g = torch.Generator().manual_seed(100 + pidx)
        e = (torch.rand(F, 2, B_folds, P_perms, generator=g) > (0.4 + 0.1 * pidx)).bool()
        edges_per_param.append(e)
    edges_5d = torch.stack(edges_per_param, dim=2)  # [F,2,P_params,B_folds,perms]

    fb = build_fold_batch(X_full, y_full, cov_full, splits)

    model = LinearCPM(edges=edges_5d, device='cpu', task_type=task_type)
    model.fit_batched(fb.X_train, fb.y_train, fb.cov_train, valid_mask=fb.train_valid)
    preds_batched = model.predict_batched(fb.X_test, fb.cov_test, valid_mask=fb.test_valid, return_proba=True)

    for pidx in range(B_params):
        for b, (tr, te) in enumerate(splits):
            Xtr, ytr, covtr = X_full[torch.as_tensor(tr)], y_full[torch.as_tensor(tr)], cov_full[torch.as_tensor(tr)]
            Xte, covte = X_full[torch.as_tensor(te)], cov_full[torch.as_tensor(te)]
            edges_ref = edges_per_param[pidx][:, :, b, :]

            ref_model = LinearCPM(edges=edges_ref, device='cpu', task_type=task_type)
            ref_model.fit(Xtr, ytr, covtr)
            ref_pred = ref_model.predict(Xte, covte, return_proba=True)

            got_pred = preds_batched[:len(te), :, :, pidx, b, :]
            np.testing.assert_allclose(got_pred.numpy(), ref_pred.numpy(), atol=1e-3)


def test_linear_cpm_batched_single_fold_single_param_matches_plain_fit():
    """B_params=B_folds=1 (the degenerate/worst-case batch plan) must
    reproduce the plain fit()/predict() path exactly."""
    torch.manual_seed(12)
    N, F, P = 30, 6, 2
    X = torch.randn(N, F)
    y = torch.randn(N, P)
    cov = torch.randn(N, 2)
    edges3d = (torch.rand(F, 2, P) > 0.4).bool()

    ref = LinearCPM(edges=edges3d, device='cpu', task_type=TaskType.regression)
    ref.fit(X, y, cov)
    ref_pred = ref.predict(X, cov, return_proba=True)

    edges5d = edges3d.unsqueeze(2).unsqueeze(3)  # [F,2,1,1,P]
    model = LinearCPM(edges=edges5d, device='cpu', task_type=TaskType.regression)
    model.fit_batched(X.unsqueeze(0), y.unsqueeze(0), cov.unsqueeze(0))
    pred_b = model.predict_batched(X.unsqueeze(0), cov.unsqueeze(0), return_proba=True)

    np.testing.assert_allclose(pred_b[:, :, :, 0, 0, :].numpy(), ref_pred.numpy(), atol=1e-4)
