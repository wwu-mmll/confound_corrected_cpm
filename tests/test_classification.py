"""Classification scoring metrics: test_scoring.py. Full pipeline: test_ground_truth.py."""

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split

from cccpm.constants import TaskType, Networks, Models
from cccpm.validation import detect_task_type, validate_task_type
from cccpm.models.linear_model import LinearCPM


@pytest.mark.parametrize("y,expected", [
    (np.array([0, 1, 0, 1, 1, 0]), TaskType.classification),
    (np.array([-1, 1, -1, 1, 1, -1]), TaskType.classification),
    (np.array([1.0, 2.5, 3.7, 4.2, 5.1]), TaskType.regression),
])
def test_detect_task_type(y, expected):
    assert detect_task_type(y) == expected


def test_detect_task_type_rejects_a_constant_target():
    with pytest.raises(ValueError):
        detect_task_type(np.array([1, 1, 1, 1]))


def test_validate_task_type_accepts_the_matching_type_and_rejects_the_other():
    y = np.array([0, 1, 0, 1])
    validate_task_type(y, TaskType.classification)
    with pytest.raises(ValueError):
        validate_task_type(y, TaskType.regression)


class TestClassificationModel:
    def test_model_classification_output(self):
        N_samples, N_features, N_runs = 20, 10, 1
        edges = torch.randint(0, 2, (N_features, 2, N_runs), dtype=torch.bool)

        model = LinearCPM(edges=edges, device='cpu', task_type=TaskType.classification)

        X = np.random.randn(N_samples, N_features).astype(np.float32)
        y = np.random.randint(0, 2, (N_samples, N_runs)).astype(np.float32)
        cov = np.random.randn(N_samples, 2).astype(np.float32)

        model.fit(X, y, cov)
        proba = model.predict(X, cov, return_proba=True)

        assert (proba >= 0).all() and (proba <= 1).all(), "Probabilities should be in [0, 1]"

    def test_regression_unchanged(self):
        np.random.seed(42)
        X_train = np.random.randn(100, 3).astype(np.float32)
        y_train = (X_train @ np.array([1.0, -0.5, 0.3]) + np.random.randn(100) * 0.1).astype(np.float32)

        edges = torch.zeros(3, 2, 1, dtype=torch.bool)
        edges[:, Networks.positive, 0] = True

        model = LinearCPM(edges=edges, device='cpu', task_type=TaskType.regression)
        cov = np.zeros((100, 1), dtype=np.float32)

        model.fit(X_train, y_train.reshape(-1, 1), cov)
        preds = model.predict(X_train, cov)

        preds_np = preds[:, Models.connectome, Networks.positive, 0].numpy()
        assert preds_np.max() > 1.0 or preds_np.min() < 0.0, (
            "Regression predictions seem sigmoid-squashed — they should be unbounded"
        )


@pytest.fixture
def binary_classification_data():
    X, y = make_classification(
        n_samples=200, n_features=1, n_informative=1, n_redundant=0,
        n_clusters_per_class=1, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def multi_feature_data():
    X, y = make_classification(
        n_samples=500, n_features=5, n_informative=3, n_redundant=1,
        n_clusters_per_class=1, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def _fit_cpm_logistic(X_train, y_train, X_test):
    """LinearCPM in classification mode, all edges, no covariates; returns proba on X_test."""
    n_features = X_train.shape[1]

    edges = torch.zeros(n_features, 2, 1, dtype=torch.bool)
    edges[:, Networks.positive, 0] = True

    model = LinearCPM(edges=edges, device='cpu', task_type=TaskType.classification)

    y_train_2d = y_train.reshape(-1, 1).astype(np.float32)
    cov_train = np.zeros((X_train.shape[0], 1), dtype=np.float32)
    cov_test = np.zeros((X_test.shape[0], 1), dtype=np.float32)

    model.fit(X_train, y_train_2d, cov_train)

    proba = model.predict_proba(X_test, cov_test)
    proba_pos = proba[:, Models.connectome, Networks.positive, 0].numpy()
    return proba_pos


def _fit_cpm_logistic_full(X_train, y_train, X_test):
    """Like _fit_cpm_logistic, but also returns the model for coefficient inspection."""
    n_features = X_train.shape[1]
    edges = torch.zeros(n_features, 2, 1, dtype=torch.bool)
    edges[:, Networks.positive, 0] = True

    model = LinearCPM(edges=edges, device='cpu', task_type=TaskType.classification)
    y_train_2d = y_train.reshape(-1, 1).astype(np.float32)
    cov_train = np.zeros((X_train.shape[0], 1), dtype=np.float32)
    cov_test = np.zeros((X_test.shape[0], 1), dtype=np.float32)

    model.fit(X_train, y_train_2d, cov_train)
    proba = model.predict_proba(X_test, cov_test)
    proba_pos = proba[:, Models.connectome, Networks.positive, 0].numpy()
    return proba_pos, model


def _fit_sklearn_logistic(X_train, y_train, X_test, penalty=None):
    clf = LogisticRegression(max_iter=1000, random_state=42, penalty=penalty, solver='lbfgs')
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    return proba, clf


class TestLogisticRegressionVsSklearn:
    def test_predicted_probabilities_close(self, binary_classification_data):
        X_train, X_test, y_train, y_test = binary_classification_data

        cpm_proba = _fit_cpm_logistic(X_train, y_train, X_test)
        sklearn_proba, _ = _fit_sklearn_logistic(X_train, y_train, X_test)

        correlation = np.corrcoef(cpm_proba, sklearn_proba)[0, 1]
        mae = np.mean(np.abs(cpm_proba - sklearn_proba))

        assert correlation > 0.999, (
            f"Predicted probabilities poorly correlated: r={correlation:.4f}"
        )
        assert mae < 0.02, (
            f"Mean absolute probability difference too large: {mae:.4f}"
        )

    def test_coefficients_match(self, binary_classification_data):
        X_train, X_test, y_train, y_test = binary_classification_data

        _, cpm_model = _fit_cpm_logistic_full(X_train, y_train, X_test)
        _, sklearn_model = _fit_sklearn_logistic(X_train, y_train, X_test)

        cpm_beta = cpm_model.coefs['connectome_positive'].squeeze().numpy()
        cpm_intercept = cpm_beta[0]
        cpm_coef = cpm_beta[1:]

        sklearn_intercept = sklearn_model.intercept_[0]
        sklearn_coef = sklearn_model.coef_[0]

        np.testing.assert_allclose(cpm_coef, sklearn_coef, rtol=0.05, atol=0.1,
            err_msg="Coefficients differ significantly between CPM IRLS and sklearn")
        np.testing.assert_allclose(cpm_intercept, sklearn_intercept, rtol=0.05, atol=0.1,
            err_msg="Intercepts differ significantly between CPM IRLS and sklearn")

    def test_multi_feature_probabilities(self, multi_feature_data):
        """CPM sums features into network strengths before fitting; sklearn gets the same summed input."""
        X_train, X_test, y_train, y_test = multi_feature_data

        X_train_summed = X_train.sum(axis=1, keepdims=True)
        X_test_summed = X_test.sum(axis=1, keepdims=True)

        cpm_proba = _fit_cpm_logistic(X_train, y_train, X_test)
        sklearn_proba, _ = _fit_sklearn_logistic(X_train_summed, y_train, X_test_summed)

        correlation = np.corrcoef(cpm_proba, sklearn_proba)[0, 1]
        mae = np.mean(np.abs(cpm_proba - sklearn_proba))

        assert correlation > 0.999, (
            f"Multi-feature probability correlation too low: r={correlation:.4f}"
        )
        assert mae < 0.02, (
            f"Mean absolute probability difference too large: {mae:.4f}"
        )

    def test_class_predictions(self, binary_classification_data):
        X_train, X_test, y_train, y_test = binary_classification_data
        n_features = X_train.shape[1]

        edges = torch.zeros(n_features, 2, 1, dtype=torch.bool)
        edges[:, Networks.positive, 0] = True

        model = LinearCPM(edges=edges, device='cpu', task_type=TaskType.classification)
        y_train_2d = y_train.reshape(-1, 1).astype(np.float32)
        cov_train = np.zeros((X_train.shape[0], 1), dtype=np.float32)
        cov_test = np.zeros((X_test.shape[0], 1), dtype=np.float32)

        model.fit(X_train, y_train_2d, cov_train)

        preds = model.predict_class(X_test, cov_test)
        preds_np = preds[:, Models.connectome, Networks.positive, 0].numpy()

        assert set(np.unique(preds_np)).issubset({0.0, 1.0}), (
            f"Class predictions should be 0 or 1, got: {np.unique(preds_np)}"
        )

        proba = model.predict_proba(X_test, cov_test)
        proba_np = proba[:, Models.connectome, Networks.positive, 0].numpy()
        expected = (proba_np > 0.5).astype(float)
        np.testing.assert_array_equal(preds_np, expected)
