"""Enums and lookup tables shared across the CCCPM pipeline."""
from enum import IntEnum, Enum


class TaskType(str, Enum):
    """Type of prediction task."""
    regression = "regression"
    classification = "classification"


class Networks(IntEnum):
    """Network types for edge selection."""
    positive = 0
    negative = 1
    both = 2


class EdgeSignificance(str, Enum):
    """Method for testing edge-stability significance under permutation: ``nbs`` (connected-component) or ``tfce`` (per-edge, no primary threshold)."""
    nbs = "nbs"
    tfce = "tfce"


class Models(IntEnum):
    """Model types in CPM analysis. Whether the connectome is deconfounded is `CPMAnalysis(model_input=...)`, not a separate model (non-linear backends aren't invariant to it)."""
    connectome = 0
    covariates = 1
    full = 2
    increment = 3


class Metrics(IntEnum):
    """Metrics for model evaluation (0-3 regression, 4-7 classification)."""
    explained_variance_score = 0
    pearson_score = 1
    mean_squared_error = 2
    mean_absolute_error = 3

    accuracy = 4
    balanced_accuracy = 5
    f1_score = 6
    roc_auc = 7


REGRESSION_METRICS = [
    Metrics.explained_variance_score,
    Metrics.pearson_score,
    Metrics.mean_squared_error,
    Metrics.mean_absolute_error
]

CLASSIFICATION_METRICS = [
    Metrics.accuracy,
    Metrics.balanced_accuracy,
    Metrics.f1_score,
    Metrics.roc_auc
]

# Metrics for which `increment` (full - covariates) is a meaningful number.
# Excluded (NaN instead): Pearson r (a difference of correlations is not a
# standard statistic -- needs Fisher z/Steiger's test, not subtraction) and
# F1 (a harmonic mean whose difference has no established interpretation).
INCREMENTABLE_METRICS = [
    Metrics.explained_variance_score,
    Metrics.mean_squared_error,
    Metrics.mean_absolute_error,
    Metrics.accuracy,
    Metrics.balanced_accuracy,
    Metrics.roc_auc,
]

METRIC_NAMES = {
    Metrics.explained_variance_score: "Explained Variance",
    Metrics.pearson_score: "Pearson r",
    Metrics.mean_squared_error: "MSE",
    Metrics.mean_absolute_error: "MAE",
    Metrics.accuracy: "Accuracy",
    Metrics.balanced_accuracy: "Balanced Accuracy",
    Metrics.f1_score: "F1 Score",
    Metrics.roc_auc: "ROC AUC"
}


def get_metrics_for_task(task_type: TaskType):
    """Returns the metric enum list for a task type."""
    if task_type == TaskType.regression:
        return REGRESSION_METRICS
    elif task_type == TaskType.classification:
        return CLASSIFICATION_METRICS
    else:
        raise ValueError(f"Unknown task type: {task_type}")
