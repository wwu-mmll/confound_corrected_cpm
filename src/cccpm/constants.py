from enum import IntEnum, Enum, auto


class TaskType(str, Enum):
    """Type of prediction task."""
    regression = "regression"
    classification = "classification"


class Networks(IntEnum):
    """Network types for edge selection."""
    positive = 0
    negative = 1
    both = 2


class Models(IntEnum):
    """Model types in CPM analysis."""
    connectome = 0
    covariates = 1
    full = 2
    residuals = 3
    increment = 4


class Metrics(IntEnum):
    """
    Metrics for model evaluation.

    Indices 0-3 are for regression tasks.
    Indices 4-7 are for classification tasks.
    """
    # Regression metrics
    explained_variance_score = 0
    pearson_score = 1
    mean_squared_error = 2
    mean_absolute_error = 3

    # Classification metrics
    accuracy = 4
    balanced_accuracy = 5
    f1_score = 6
    roc_auc = 7


# Metric groups for different task types
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

# Metric names for display
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
    """
    Get the appropriate metrics for a given task type.

    Args:
        task_type: Type of task (regression or classification)

    Returns:
        List of metric enum values
    """
    if task_type == TaskType.regression:
        return REGRESSION_METRICS
    elif task_type == TaskType.classification:
        return CLASSIFICATION_METRICS
    else:
        raise ValueError(f"Unknown task type: {task_type}")
