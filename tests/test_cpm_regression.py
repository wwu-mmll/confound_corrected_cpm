import pytest
import numpy as np
import pandas as pd
from cccpm.utils import check_data

# The 'cpm_instance' and 'simulated_data' fixtures come from conftest.py

def test_run(cpm_instance, simulated_data):
    X, y, covariates = simulated_data
    # We use the fixture instance which is already configured with a tmp_path
    cpm_instance.run(X, y, covariates)

def test_input_is_dataframe(cpm_instance, simulated_data):
    X, y, covariates = simulated_data
    cpm_instance.run(
        pd.DataFrame(X),
        pd.DataFrame(y),
        pd.DataFrame(covariates)
    )

# Missing Values Tests
def test_nan_in_X(simulated_data):
    X, y, covariates = simulated_data
    # Work on a copy to avoid affecting other tests reusing the fixture
    X_nan = X.copy()
    X_nan[0, 0] = np.nan

    with pytest.raises(ValueError):
        check_data(X_nan, y, covariates, impute_missings=False)

    # Should not raise
    check_data(X_nan, y, covariates, impute_missings=True)

def test_nan_in_y(simulated_data):
    X, y, covariates = simulated_data
    y_nan = y.copy()
    y_nan[0] = np.nan

    # raise error if y contains nan and impute_missings is False
    with pytest.raises(ValueError):
        check_data(X, y_nan, covariates, impute_missings=False)

    # but also raise an error if y contains nan and impute_missings is True
    # values in y should never be missing
    with pytest.raises(ValueError):
        check_data(X, y_nan, covariates, impute_missings=True)