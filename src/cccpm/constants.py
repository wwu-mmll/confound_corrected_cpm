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
    """Method for testing edge-stability significance under permutation.

    ``nbs``  — Network-Based Statistic: connected-component test controlling
               FWER via a permutation max-component null (subnetwork-level).
    ``tfce`` — Threshold-Free Cluster Enhancement: per-edge FWER control with
               no arbitrary primary threshold.
    """
    nbs = "nbs"
    tfce = "tfce"


class Models(IntEnum):
    """
    Model types in CPM analysis.

    Whether the connectome fed to these models has been deconfounded is a
    property of the run (``CPMAnalysis(model_input=...)``), not a separate
    model. See that parameter for why: the non-linear backends are not
    invariant to it, so it cannot be a model name without meaning different
    things for different backends.
    """
    connectome = 0
    covariates = 1
    full = 2
    increment = 3


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

# Metrics for which `increment` (full - covariates) is a meaningful number.
#
# `increment` is a *difference of two metrics*, which is only interpretable when
# differences of that quantity are themselves a standard statistic:
#
#   explained variance   difference of R^2 -- the squared semipartial correlation
#   MSE / MAE            error reduction (negative when the connectome helps)
#   accuracy             difference of proportions
#   balanced accuracy    difference of a mean of two proportions
#   ROC AUC              difference of AUCs (as in a DeLong comparison)
#
# Excluded:
#
#   Pearson r            a difference of two correlations is not a standard
#                        statistic; comparing correlations needs Fisher z or
#                        Steiger's test, not subtraction
#   F1                   a harmonic mean of precision and recall, whose
#                        difference has no established interpretation
#
# The excluded slots are written as NaN rather than a plausible-looking number.
INCREMENTABLE_METRICS = [
    Metrics.explained_variance_score,
    Metrics.mean_squared_error,
    Metrics.mean_absolute_error,
    Metrics.accuracy,
    Metrics.balanced_accuracy,
    Metrics.roc_auc,
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
