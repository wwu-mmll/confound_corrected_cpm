from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt


def generate_scenario_data(scenario, n_samples=1000, n_features=10, n_informative=3, noise_level=0.1):
    np.random.seed(42)  # For reproducibility

    # Generate feature matrix X
    X = np.random.normal(0, 1, (n_samples, n_features))

    # Generate ground truth coefficients for X's influence on z and y
    z_coefficients = np.zeros(n_features)
    y_coefficients = np.zeros(n_features)

    if scenario == 1:
        # Scenario 1: y and z are influenced by X independently
        z_coefficients[:n_informative] = np.linspace(1, 0.1, n_informative)  # Decreasing influence
        y_coefficients[:n_informative] = np.linspace(1, 0.1, n_informative)  # Decreasing influence
        z = np.dot(X, z_coefficients) + np.random.normal(0, noise_level, n_samples)
        y = np.dot(X, y_coefficients) + np.random.normal(0, noise_level, n_samples)

    elif scenario == 2:
        # Scenario 2: z influences both X and y, inducing spurious association
        z = np.random.normal(0, 1, n_samples)
        z_coefficients[:n_informative] = np.linspace(1, 0.1, n_informative)
        y = 2 * z + np.random.normal(0, noise_level, n_samples)
        for i in range(n_features):
            X[:, i] = z * z_coefficients[i] + np.random.normal(0, noise_level, n_samples)

    elif scenario == 3:
        # Scenario 3: y is influenced by both X and z, with z partially mediating
        z_coefficients[:n_informative] = np.linspace(1, 0.1, n_informative)
        y_coefficients[:n_informative] = np.linspace(1, 0.1, n_informative)
        z = np.dot(X, z_coefficients) + np.random.normal(0, noise_level, n_samples)
        y = np.dot(X, y_coefficients) + 0.5 * z + np.random.normal(0, noise_level, n_samples)
    elif scenario == 4:
        y_coefficients[:n_informative] = np.linspace(1, 0.1, n_informative)
        z = np.random.normal(0, noise_level, n_samples)
        y = np.dot(X, y_coefficients) + np.random.normal(0, noise_level, n_samples)

    else:
        raise ValueError("Invalid scenario. Choose 1, 2, or 3.")

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


# Generate data for each scenario and calculate explained variances
for scenario in range(1, 5):
    X, y, z = generate_scenario_data(scenario, noise_level=1)

    # Calculate explained variances
    explained_variance_X_y = calculate_explained_variance(X, y)
    explained_variance_X_z = calculate_explained_variance(X, z)
    explained_variance_y_given_z = calculate_explained_variance_y_given_z(y, z)

    print(f"Scenario {scenario}:")
    print(f"  Explained Variance (R^2) of X for y: {explained_variance_X_y:.2f}")
    print(f"  Explained Variance (R^2) of X for z: {explained_variance_X_z:.2f}")
    print(f"  Explained Variance (R^2) of y for z: {explained_variance_y_given_z:.2f}")
    print(f"  Corr(X[:, 0], y): {np.corrcoef(X[:, 0], y)[0, 1]:.2f}")
    print(f"  Corr(X[:, 0], z): {np.corrcoef(X[:, 0], z)[0, 1]:.2f}")
    print(f"  Corr(y, z): {np.corrcoef(y, z)[0, 1]:.2f}")

    # Plot y vs z for visualization
    plt.figure()
    plt.scatter(y, z, alpha=0.5, label="y vs z")
    plt.title(f"Scenario {scenario}: y vs z")
    plt.xlabel("y")
    plt.ylabel("z")
    plt.legend()
    plt.show()
