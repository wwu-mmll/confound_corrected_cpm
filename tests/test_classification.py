import pytest
import torch
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from cccpm.constants import TaskType, Metrics
from cccpm.utils import detect_task_type, validate_task_type
from cccpm.scoring import FastCPMClassificationMetrics, score_models
from cccpm.pytorch_model import LinearCPMModel


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


class TestClassificationScoring:
    @pytest.fixture
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_perfect_classification(self, device):
        """Test that perfect predictions yield accuracy/balanced_accuracy/f1 of 1."""
        N_samples = 50
        N_runs = 1

        y_true = torch.cat([torch.zeros(25), torch.ones(25)]).unsqueeze(1)
        # Perfect probabilities: 0 for class 0, 1 for class 1
        y_pred = y_true.clone()
        # Expand to [N_samples, N_models, N_networks, N_runs]
        y_pred_full = y_pred.unsqueeze(1).unsqueeze(1).expand(-1, 5, 3, -1)

        evaluator = FastCPMClassificationMetrics(device=device)
        scores = evaluator.score(y_true.numpy(), y_pred_full.numpy())

        assert torch.allclose(scores[Metrics.accuracy], torch.ones_like(scores[Metrics.accuracy]), atol=1e-5)
        assert torch.allclose(scores[Metrics.balanced_accuracy], torch.ones_like(scores[Metrics.balanced_accuracy]), atol=1e-5)
        assert torch.allclose(scores[Metrics.f1_score], torch.ones_like(scores[Metrics.f1_score]), atol=1e-5)
        assert torch.allclose(scores[Metrics.roc_auc], torch.ones_like(scores[Metrics.roc_auc]), atol=1e-5)

    def test_score_models_classification(self, device):
        """Test that score_models routes correctly for classification."""
        N_samples = 40
        N_runs = 1

        y_true = np.concatenate([np.zeros(20), np.ones(20)]).reshape(-1, 1)
        y_pred = torch.rand(N_samples, 5, 3, N_runs)

        scores = score_models(y_true, y_pred, task_type=TaskType.classification, device=device)
        assert scores.shape[0] == len(Metrics)

        # Classification metrics should be populated
        acc = scores[Metrics.accuracy]
        assert (acc >= 0).all() and (acc <= 1).all()

    def test_output_shape(self, device):
        N_samples, N_runs = 30, 2
        y_true = np.random.randint(0, 2, (N_samples, N_runs)).astype(np.float32)
        y_pred = torch.rand(N_samples, 5, 3, N_runs)

        evaluator = FastCPMClassificationMetrics(device=device)
        scores = evaluator.score(y_true, y_pred)

        assert scores.shape == (len(Metrics), 5, 3, N_runs)


class TestClassificationModel:
    @pytest.fixture
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_model_classification_output(self, device):
        """Test that classification model outputs probabilities in [0, 1]."""
        N_samples, N_features, N_runs = 20, 10, 1
        edges = torch.randint(0, 2, (N_features, 2, N_runs), dtype=torch.bool)

        model = LinearCPMModel(edges=edges, device=device, task_type=TaskType.classification)

        X = np.random.randn(N_samples, N_features).astype(np.float32)
        y = np.random.randint(0, 2, (N_samples, N_runs)).astype(np.float32)
        cov = np.random.randn(N_samples, 2).astype(np.float32)

        model.fit(X, y, cov)
        proba = model.predict(X, cov, return_proba=True)

        assert (proba >= 0).all() and (proba <= 1).all(), "Probabilities should be in [0, 1]"

    def test_model_predict_class(self, device):
        """Test that predict_class returns 0/1."""
        N_samples, N_features, N_runs = 20, 10, 1
        edges = torch.randint(0, 2, (N_features, 2, N_runs), dtype=torch.bool)

        model = LinearCPMModel(edges=edges, device=device, task_type=TaskType.classification)

        X = np.random.randn(N_samples, N_features).astype(np.float32)
        y = np.random.randint(0, 2, (N_samples, N_runs)).astype(np.float32)
        cov = np.random.randn(N_samples, 2).astype(np.float32)

        model.fit(X, y, cov)
        classes = model.predict_class(X, cov)

        unique_vals = torch.unique(classes)
        assert all(v in [0.0, 1.0] for v in unique_vals.tolist()), "Class predictions should be 0 or 1"


class TestClassificationPipeline:
    def test_full_classification_run(self, cpm_classification_instance, simulated_classification_data):
        """Test that a full classification CPM pipeline runs without errors."""
        X, y, covariates = simulated_classification_data
        cpm_classification_instance.run(X, y, covariates)

        # Verify task type was set
        assert cpm_classification_instance.task_type == TaskType.classification

        # Verify results exist
        assert cpm_classification_instance.results_manager is not None
        assert cpm_classification_instance.results_manager.agg_results is not None
