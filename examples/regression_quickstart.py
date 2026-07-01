"""
Quickstart: Connectome-based Predictive Modeling for a *regression* target.

This script runs a complete CPM analysis on simulated data in a few lines:
it selects predictive edges, fits the models inside a cross-validation loop,
assesses significance with permutation testing, and writes an HTML report.

Run it with:

    poetry run python examples/regression_quickstart.py

The companion script ``classification_quickstart.py`` shows the same workflow
for a binary target.
"""

import numpy as np
from sklearn.model_selection import KFold

from cccpm import CPMAnalysis, UnivariateEdgeSelection, PThreshold
from cccpm.simulation.simulate_sem import simulate_data_given_kappa

# ---------------------------------------------------------------------------
# 1. Get some data
# ---------------------------------------------------------------------------
# In a real study these would be your own arrays:
#   X          : connectome edges, shape (n_samples, n_features)
#   y          : the (continuous) outcome you want to predict, shape (n_samples,)
#   covariates : nuisance variables to control for (e.g. age, motion, sex),
#                shape (n_samples, n_covariates)
#
# Here we use the SEM-based simulator, which builds data with a *known* ground
# truth: a latent confound Z is a common cause of both the connectome and the
# outcome. ``kappa`` is the fraction of the naive brain–outcome R² that is
# actually confound-driven, so ``kappa=0.3`` means 30% of the apparent signal is
# spurious and confound control genuinely matters.
sim = simulate_data_given_kappa(
    R2_X_y=0.4,          # naive R²(y ~ X): apparent brain–outcome strength
    kappa=0.3,           # 30% of that R² is driven by the confound
    n_features=435,      # a 30-node connectome (30*29/2 = 435 edges)
    n_features_informative=40,    # "mixed" edges: real signal + confound leakage
    n_pure_signal_features=20,    # edges tied to y but NOT the confound
    n_confound_only_features=20,  # edges tied to y ONLY through the confound
    n_confounds=2,
    n_samples=200,
    random_state=42,
)
X, y, covariates = sim["X"], sim["y"], sim["Z"]

# The simulator records the analytic ground truth, so we know what "good"
# deconfounded performance should look like before we even run the model.
info = sim["info"]
print(f"Naive R²(y~X)          : {info['R2_X_y']:.2f}  (inflated by the confound)")
print(f"True R²(y~X | Z)        : {info['R2_X_y_given_Z']:.2f}  (what we hope to recover)")

# ---------------------------------------------------------------------------
# 2. Configure edge selection
# ---------------------------------------------------------------------------
# Pick which edges enter the model by correlating each edge with the target and
# keeping those below a p-value threshold. Use 'pearson_partial' instead of
# 'pearson' to control for the covariates already *during* edge selection.
edge_selection = UnivariateEdgeSelection(
    edge_statistic="pearson",
    edge_selection=[PThreshold(threshold=[0.05], correction=[None])],
)

# ---------------------------------------------------------------------------
# 3. Configure and run the analysis
# ---------------------------------------------------------------------------
cpm = CPMAnalysis(
    results_directory="./results/regression_quickstart",
    task_type="regression",                       # or leave as None to auto-detect
    cv=KFold(n_splits=10, shuffle=True, random_state=42),
    edge_selection=edge_selection,
    n_permutations=1000,                           # use 1000+ for a real analysis
    device="cpu",                                 # "cuda" uses the GPU if available
    edge_significance_method='nbs',
    nbs_threshold=0.5
)

cpm.run(X=X, y=y, covariates=covariates)

# ---------------------------------------------------------------------------
# 4. Inspect the results
# ---------------------------------------------------------------------------
# Everything is written under the results_directory, including:
#   - cv_results_summary.csv : performance metrics (Pearson r, MAE, MSE, ...)
#   - p_values.csv           : permutation-based significance
#   - cv_predictions.csv     : out-of-sample predictions per subject
#   - report.html            : a full, human-readable HTML report
#
# Because the connectome carries genuine confound leakage, compare the 'connectome'
# model against the 'residuals' model (which removes the covariates first): the gap
# between them is the confound inflation the SEM simulator built in on purpose.
print("Done. Open ./results/regression_quickstart/report.html to explore the results.")
