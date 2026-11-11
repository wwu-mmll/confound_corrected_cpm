"""Per-fold data preparation: anything estimated from the data is fit on train and only applied to test."""
import torch


def to_device(array, device):
    """Copies array into a float32 tensor on device.

    Always a copy, even when ``torch.as_tensor`` would alias memory --
    the run's working tensor must never alias the caller's own data (e.g.
    a read-only pandas copy-on-write view).
    """
    return torch.tensor(array, dtype=torch.float32, device=device)


def torch_train_test_split(train, test, X, y, covariates):
    """Splits X/y/covariates into train/test by index (tensor version of ``utils.train_test_split``)."""
    train_idx = torch.as_tensor(train, device=X.device)
    test_idx = torch.as_tensor(test, device=X.device)
    return (X[train_idx], X[test_idx], y[train_idx], y[test_idx],
            covariates[train_idx], covariates[test_idx])


def torch_impute_missing_values(X_train, X_test, cov_train, cov_test):
    """Mean-imputes NaNs in X and covariates, using column means fit on train."""
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
    """Regresses X ~ intercept + confounds via OLS fit on train, subtracts fitted values from both splits."""
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
    """Thresholds per-edge selection stability (fraction of folds selected) into a boolean edge mask."""
    return stability_edges > stability_threshold
