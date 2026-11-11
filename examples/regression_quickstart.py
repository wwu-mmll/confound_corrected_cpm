"""Quickstart: Connectome-based Predictive Modeling for a regression target.

Run it with:

    poetry run python examples/regression_quickstart.py
"""

from sklearn.model_selection import KFold, ShuffleSplit

from cccpm import CPMAnalysis, UnivariateEdgeSelection, PThreshold
from cccpm.simulation.simulate_sem import simulate_data_given_kappa

# The SEM-based simulator builds data with a known ground truth: a latent
# confound Z is a common cause of both the connectome and the outcome. ``kappa``
# is the fraction of the naive brain-outcome R^2 that is actually confound-driven.
sim = simulate_data_given_kappa(
    R2_X_y=0.4,
    kappa=0.3,
    n_features=4950,  # matches the built-in Schaefer100 atlas used below
    n_features_informative=40,
    n_pure_signal_features=20,
    n_confound_only_features=20,
    n_confounds=2,
    n_samples=200,
    random_state=42,
)
X, y, covariates = sim["X"], sim["y"], sim["Z"]

info = sim["info"]
print(f"Naive R²(y~X)          : {info['R2_X_y']:.2f}  (inflated by the confound)")
print(f"True R²(y~X | Z)        : {info['R2_X_y_given_Z']:.2f}  (what we hope to recover)")

# Confound control is two independent choices. `selection_input` (here) decides
# whether edge *selection* controls for the covariates; `model_input` (below) is
# the other. Several thresholds turn PThreshold into a tuned hyperparameter,
# picked per outer fold by the inner CV so it never sees the outer test set.
edge_selection = UnivariateEdgeSelection(
    selection_statistic="pearson",
    selection_input="residualized",
    edge_selection=[PThreshold(threshold=[0.05, 0.01, 0.001], correction=[None])],
)

cpm = CPMAnalysis(
    results_directory="./results/regression_quickstart",
    task_type="regression",
    cv=KFold(n_splits=10, shuffle=True, random_state=42),
    inner_cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
    edge_selection=edge_selection,
    model_input="residualized",
    n_permutations=1000,
    atlas="Schaefer100-17",
    device="cpu",
    stability_significance_method="nbs",
    nbs_stability_threshold=0.5,
)

cpm.run(X=X, y=y, covariates=covariates)

# To see the confound inflation the simulator built in, rerun with
# selection_input='raw' and model_input='raw' and compare 'connectome' across
# the two runs -- the gap is the inflation.
print("Done. Open ./results/regression_quickstart/report.html to explore the results.")
