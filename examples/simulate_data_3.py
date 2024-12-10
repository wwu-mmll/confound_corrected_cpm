import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, ShuffleSplit, RepeatedKFold
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score
from cpm import CPMRegression
from cpm.simulate_data import simulate_regression_data_2
from cpm.edge_selection import PThreshold, UnivariateEdgeSelection


def generate_scenario_data(scenario, n_samples=1000, n_features=10, n_informative=3, noise_level=0.1):
    np.random.seed(42)  # For reproducibility

    # Generate feature matrix X
    X = np.random.normal(0, 1, (n_samples, n_features))
    z = np.random.normal(0, 1, n_samples)
    y = np.random.normal(0, 1, n_samples)

    # Generate ground truth coefficients for X's influence on z and y
    z_coefficients = np.empty(n_features)
    y_coefficients = np.empty(n_features)

    z_coefficients[:n_informative]  = np.linspace(0.3, 0.1, n_informative)
    y_coefficients[:n_informative]  = np.linspace(0.3, 0.1, n_informative)

    if scenario == "A1":
        pass

    elif scenario == "A2":
        for i in range(n_features):
            X[:, i] = X[:, i] + 0.8 * z * z_coefficients[i]

    elif scenario == "A3":
        y = y + 0.8 * z

    elif scenario == "A4":
        for i in range(n_features):
            X[:, i] = X[:, i] + 0.8 * z * z_coefficients[i]
        y = y + 0.8 * z

    elif scenario == "B1":
        for i in range(n_features):
            X[:, i] =  X[:, i] + y * y_coefficients[i]

    elif scenario == "B2":
        for i in range(n_features):
            X[:, i] = X[:, i] + y * y_coefficients[i]
        model = LinearRegression()
        model.fit(y.reshape(-1, 1), X[:, 0])
        resid = X[:, 0] - model.predict(y.reshape(-1, 1))
        z = z + 2 * resid

    elif scenario == "B3":
        for i in range(n_features):
            X[:, i] = X[:, i] + y * y_coefficients[i]
        model = LinearRegression()
        model.fit(X[:, 0].reshape(-1, 1), y)
        resid = y - model.predict(X[:, 0].reshape(-1, 1))
        z = z + 2 * resid

    elif scenario == "B4":
        z = 1.5 * z + y
        for i in range(n_features):
            X[:, i] = X[:, i] + 2 * y * y_coefficients[i] + 0.1 * z * z_coefficients[i]

    else:
        raise NotImplementedError("Invalid scenario.")

    return X, y, z


def calculate_explained_variance(X, target):
    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X, target)

    # Predict the target using X
    target_pred = model.predict(X)

    # Calculate R^2 (explained variance)
    r2 = r2_score(target, target_pred)
    return r2


def calculate_explained_variance_y_given_z(y, z):
    # Fit a linear regression model
    model = LinearRegression()
    z = z.reshape(-1, 1)  # Reshape z to fit as a single feature
    model.fit(z, y)
    y_pred = model.predict(z)
    r2 = r2_score(y, y_pred)
    return r2


def residualize_X(X, z):
    """Regress out the effect of z from each column of X."""
    z = z.reshape(-1, 1)  # Reshape z for regression
    residualized_X = np.zeros_like(X)

    for i in range(X.shape[1]):  # Iterate over columns (features)
        model = LinearRegression()
        model.fit(z, X[:, i])
        # Compute residuals
        residualized_X[:, i] = X[:, i] - model.predict(z)

    return residualized_X


def calculate_explained_variance_y_with_residualized_X(X, y, z):
    """Calculate explained variance of y by X after removing z's influence on X."""
    X_residualized = residualize_X(X, z)
    return calculate_explained_variance(X_residualized, y)


#edge_statistics = ['pearson', 'pearson_partial']
edge_statistics = ['pearson']

# Generate data for each scenario and calculate explained variances
for scenario in ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]:
    for edge_statistic in edge_statistics:
        print(f"Scenario {scenario}:")
        X, y, z = generate_scenario_data(scenario, noise_level=1, n_samples=1000, n_features=105, n_informative=10)
        folder = f"simulated_data/scenario_{scenario}"
        os.makedirs(folder, exist_ok=True)
        np.save(f"{folder}/X.npy", X)
        np.save(f"{folder}/y.npy", y)
        np.save(f"{folder}/covariates.npy", z)


        univariate_edge_selection = UnivariateEdgeSelection(edge_statistic=[edge_statistic],
                                                            edge_selection=[PThreshold(threshold=[0.05],
                                                                                       correction=['fdr_by'])])
        cpm = CPMRegression(results_directory=os.path.join(folder, 'results', f'{edge_statistic}'),
                            cv=RepeatedKFold(n_splits=10, n_repeats=5, random_state=42),
                            edge_selection=univariate_edge_selection,
                            # cv_edge_selection=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
                            add_edge_filter=True,
                            n_permutations=2)
        cpm.estimate(X=X, y=y, covariates=z.reshape(-1, 1))

        # Calculate explained variances
        explained_variance_X_y = calculate_explained_variance(X[:,0].reshape(-1, 1), y)
        explained_variance_X_z = calculate_explained_variance(X[:,0].reshape(-1, 1), z)
        explained_variance_y_given_z = calculate_explained_variance_y_given_z(y, z)
        explained_variance_X_y_residualized = calculate_explained_variance_y_with_residualized_X(X[:,0].reshape(-1, 1), y, z)

        print(f"  Explained Variance (R^2) in y given X: {explained_variance_X_y:.2f}")
        print(f"  Explained Variance (R^2) in y given X controlling for z: {explained_variance_X_y_residualized:.2f}")
        print(f"  Explained Variance (R^2) in z given X: {explained_variance_X_z:.2f}")
        print(f"  Explained Variance (R^2) in y given z: {explained_variance_y_given_z:.2f}")
