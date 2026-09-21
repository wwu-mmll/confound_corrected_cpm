"""
Per-fold data preparation.

The transformations applied between splitting a fold and fitting a model:
train/test indexing on device, mean imputation fitted on train, confound
residualisation fitted on train, and thresholding edge stability into a mask.

Everything here follows the same rule -- anything estimated from the data is
estimated on the training split and only applied to the test split.
"""
import torch


def torch_train_test_split(train, test, X, y, covariates):
    """Wie utils.train_test_split, aber X/y/covariates sind bereits GPU-Tensoren."""
    train_idx = torch.as_tensor(train, device=X.device)
    test_idx = torch.as_tensor(test, device=X.device)
    return (X[train_idx], X[test_idx], y[train_idx], y[test_idx],
            covariates[train_idx], covariates[test_idx])


def torch_impute_missing_values(X_train, X_test, cov_train, cov_test):
    def _impute(train, test):
        train = torch.as_tensor(train, dtype=torch.float32)
        test = torch.as_tensor(test, dtype=torch.float32)
        col_mean = torch.nanmean(train, dim=0)
        col_mean = torch.nan_to_num(col_mean, nan=0.0)
        train_filled = torch.where(torch.isnan(train), col_mean.unsqueeze(0), train)
        test_filled = torch.where(torch.isnan(test), col_mean.unsqueeze(0), test)
        return train_filled, test_filled

    X_train, X_test = _impute(X_train, X_test)
    cov_train, cov_test = _impute(cov_train, cov_test)
    return X_train, X_test, cov_train, cov_test


def residualize_train_test(X_train, X_test, confounds_train, confounds_test):
    """
    Regress X ~ intercept + confounds via OLS on the training set only, then
    subtract the fitted values from both train and test (residualizing test
    with the train-fit model, never fitting on test data). Pure torch,
    dtype/device-agnostic (works equally on CPU or GPU tensors) -- same
    closed-form pseudo-inverse approach as statistics.get_residuals,
    generalized to the fit-on-train/apply-to-both split CPMAnalysis needs
    for its `calculate_residuals` option.

    Args:
        X_train, X_test: [N_train, F], [N_test, F].
        confounds_train, confounds_test: [N_train, C], [N_test, C].

    Returns:
        X_train_resid, X_test_resid: same shapes as X_train, X_test.
    """
    dtype = X_train.dtype
    device = X_train.device
    confounds_train = torch.as_tensor(confounds_train, dtype=dtype, device=device)
    confounds_test = torch.as_tensor(confounds_test, dtype=dtype, device=device)

    ones_train = torch.ones(confounds_train.shape[0], 1, dtype=dtype, device=device)
    ones_test = torch.ones(confounds_test.shape[0], 1, dtype=dtype, device=device)
    Z_train = torch.cat([ones_train, confounds_train], dim=1)
    Z_test = torch.cat([ones_test, confounds_test], dim=1)

    Z_pinv = torch.linalg.pinv(Z_train)
    beta = torch.matmul(Z_pinv, X_train)

    X_train_resid = X_train - torch.matmul(Z_train, beta)
    X_test_resid = X_test - torch.matmul(Z_test, beta)
    return X_train_resid, X_test_resid


def select_stable_edges(stability_edges, stability_threshold):
    """
    Threshold per-edge selection stability into a boolean edge mask.

    Args:
        stability_edges: Tensor [N_features, 2, N_runs] (fraction of folds each
                          edge was selected in; dim 1 = [Positive, Negative]),
                          as returned by ResultsManager.calculate_edge_stability.
        stability_threshold: float; edges selected in more than this fraction
                              of folds are kept.

    Returns:
        Boolean tensor [N_features, 2, N_runs], same convention as every other
        edge mask in the pipeline (e.g. PThreshold.select's return value).
    """
    return stability_edges > stability_threshold
