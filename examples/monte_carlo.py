import numpy as np
import matplotlib.pyplot as plt


def simulate_scenario(scenario, n=1000, noise_level=0.1):
    np.random.seed(42)  # For reproducibility
    if scenario == 1:
        # Scenario 1: x and y associated, z independent
        x = np.random.normal(0, 1, n)
        y = 2 * x + np.random.normal(0, noise_level, n)
        z = np.random.normal(0, 1, n)  # Independent of x and y
    elif scenario == 2:
        # Scenario 2: x and y spurious due to z
        z = np.random.normal(0, 1, n)
        x = 2 * z + np.random.normal(0, noise_level, n)
        y = -3 * z + np.random.normal(0, noise_level, n)
    elif scenario == 3:
        # Scenario 3: x and y associated; z partially explains it
        z = np.random.normal(0, 1, n)
        x = 2 * z + np.random.normal(0, noise_level, n)
        y = 3 * x - 1.5 * z + np.random.normal(0, noise_level, n)
    else:
        raise ValueError("Invalid scenario. Choose 1, 2, or 3.")
    return x, y, z


# Simulate and plot results for each scenario
for scenario in range(1, 4):
    x, y, z = simulate_scenario(scenario, noise_level=2)

    print(f"Scenario {scenario} correlations:")
    print(f"  Corr(x, y): {np.corrcoef(x, y)[0, 1]:.2f}")
    print(f"  Corr(x, z): {np.corrcoef(x, z)[0, 1]:.2f}")
    print(f"  Corr(y, z): {np.corrcoef(y, z)[0, 1]:.2f}")

    plt.figure()
    plt.scatter(x, y, alpha=0.5, label="x vs y")
    plt.scatter(x, z, alpha=0.5, label="x vs z")
    plt.scatter(y, z, alpha=0.5, label="y vs z")
    plt.legend()
    plt.title(f"Scenario {scenario}")
    plt.xlabel("Variable")
    plt.ylabel("Variable")
    plt.show()
