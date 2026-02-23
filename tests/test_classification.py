"""
Tests for classification support: task type detection, classification model outputs,
and IRLS logistic regression correctness (compared against sklearn).

Classification scoring metrics are validated in test_scoring.py.
Full classification pipeline ground-truth tests are in test_ground_truth.py.
"""

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from cccpm.constants import TaskType, Networks, Models
from cccpm.utils import detect_task_type, validate_task_type
from cccpm.models.linear_model import LinearCPM


# ============================================================
# Task type detection
# ============================================================

class TestTaskTypeDetection:
    def test_detect_binary_01(self):
        y = np.array([0, 1, 0, 1, 1, 0])
        assert detect_task_type(y) == TaskType.classification

    def test_detect_binary_neg1_1(self):
        y = np.array([-1, 1, -1, 1, 1, -1])
        assert detect_task_type(y) == TaskType.classification

    def test_detect_continuous(self):
        y = np.array([1.0, 2.5, 3.7, 4.2, 5.1])
        assert detect_task_type(y) == TaskType.regression

    def test_validate_correct_classification(self):
        y = np.array([0, 1, 0, 1])
        validate_task_type(y, TaskType.classification)

    def test_validate_wrong_type_raises(self):
        y = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError):
            validate_task_type(y, TaskType.regression)

    def test_constant_target_raises(self):
        y = np.array([1, 1, 1, 1])
        with pytest.raises(ValueError):
            detect_task_type(y)


# ============================================================
# Classification model outputs
# ============================================================

class TestClassificationModel:
    def test_model_classification_output(self):
        """Test that classification model outputs probabilities in [0, 1]."""
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
        """Verify that regression mode still uses OLS (not IRLS)."""
        np.random.seed(42)
        X_train = np.random.randn(100, 3).astype(np.float32)
        y_train = (X_train @ np.array([1.0, -0.5, 0.3]) + np.random.randn(100) * 0.1).astype(np.float32)

        edges = torch.zeros(3, 2, 1, dtype=torch.bool)
        edges[:, Networks.positive, 0] = True

        model = LinearCPM(edges=edges, device='cpu', task_type=TaskType.regression)
        cov = np.zeros((100, 1), dtype=np.float32)

        model.fit(X_train, y_train.reshape(-1, 1), cov)
        preds = model.predict(X_train, cov)

        # For regression, predictions should NOT be squashed through sigmoid
        preds_np = preds[:, Models.connectome, Networks.positive, 0].numpy()
        assert preds_np.max() > 1.0 or preds_np.min() < 0.0, (
            "Regression predictions seem sigmoid-squashed — they should be unbounded"
        )


# ============================================================
# IRLS logistic regression vs sklearn
# ============================================================

@pytest.fixture
def binary_classification_data():
    """Generate a binary classification dataset with a single informative feature."""
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
    """Generate a binary classification dataset with multiple features."""
    X, y = make_classification(
        n_samples=500, n_features=5, n_informative=3, n_redundant=1,
        n_clusters_per_class=1, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def _fit_cpm_logistic(X_train, y_train, X_test):
    """
    Fit a LinearCPM in classification mode using the 'connectome_positive' path
    (all edges selected), with no covariates. Returns predicted probabilities on X_test.
    """
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
    """Same as above but returns the model as well for coefficient inspection."""
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
    """Fit sklearn LogisticRegression (unregularized) and return predicted probabilities."""
    clf = LogisticRegression(max_iter=1000, random_state=42, penalty=penalty, solver='lbfgs')
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    return proba, clf


class TestLogisticRegressionVsSklearn:
    """Compare CPM's IRLS logistic regression with sklearn's implementation."""

    def test_predicted_probabilities_close(self, binary_classification_data):
        """Predicted probabilities from CPM IRLS and sklearn should be nearly identical."""
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

    def test_classification_accuracy_comparable(self, binary_classification_data):
        """Classification accuracy from both methods should match."""
        X_train, X_test, y_train, y_test = binary_classification_data

        cpm_proba = _fit_cpm_logistic(X_train, y_train, X_test)
        sklearn_proba, _ = _fit_sklearn_logistic(X_train, y_train, X_test)

        cpm_acc = accuracy_score(y_test, (cpm_proba > 0.5).astype(int))
        sklearn_acc = accuracy_score(y_test, (sklearn_proba > 0.5).astype(int))

        cpm_auc = roc_auc_score(y_test, cpm_proba)
        sklearn_auc = roc_auc_score(y_test, sklearn_proba)

        assert abs(cpm_acc - sklearn_acc) < 0.05, (
            f"Accuracy gap too large: CPM={cpm_acc:.4f} vs sklearn={sklearn_acc:.4f}"
        )
        assert abs(cpm_auc - sklearn_auc) < 0.01, (
            f"AUC gap too large: CPM={cpm_auc:.4f} vs sklearn={sklearn_auc:.4f}"
        )

    def test_coefficients_match(self, binary_classification_data):
        """IRLS coefficients should closely match sklearn's unregularized logistic regression (both MLE)."""
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
        """
        With multiple features, CPM sums them into network strengths before fitting.
        For a fair comparison, give sklearn the same summed input.
        """
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

    def test_probability_calibration(self, multi_feature_data):
        """IRLS should produce probability ranges comparable to sklearn's."""
        X_train, X_test, y_train, y_test = multi_feature_data

        cpm_proba = _fit_cpm_logistic(X_train, y_train, X_test)
        sklearn_proba, _ = _fit_sklearn_logistic(X_train, y_train, X_test)

        cpm_range = cpm_proba.max() - cpm_proba.min()
        sklearn_range = sklearn_proba.max() - sklearn_proba.min()

        range_ratio = cpm_range / sklearn_range if sklearn_range > 0 else 0

        assert range_ratio > 0.8, (
            f"CPM probability range much narrower than sklearn's: ratio={range_ratio:.4f}"
        )

    def test_class_predictions(self, binary_classification_data):
        """predict_class returns 0/1 matching thresholded probabilities."""
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
