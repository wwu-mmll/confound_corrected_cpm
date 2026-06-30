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
from cccpm.simulation.simulate_simple import simulate_confounded_data_chyzhyk

# ---------------------------------------------------------------------------
# 1. Get some data
# ---------------------------------------------------------------------------
# In a real study these would be your own arrays:
#   X          : connectome edges, shape (n_samples, n_features)
#   y          : the (continuous) outcome you want to predict, shape (n_samples,)
#   covariates : nuisance variables to control for (e.g. age, motion, sex),
#                shape (n_samples, n_covariates)
#
# Here we simulate data where the target is genuinely linked to the connectome
# but also entangled with a confound, so confound control actually matters.
X, y, covariates = simulate_confounded_data_chyzhyk(
    n_samples=100, n_features=435, link_type="direct_link"
)

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
    n_permutations=100,                           # use 1000+ for a real analysis
    device="cpu",                                 # "cuda" uses the GPU if available
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
print("Done. Open ./results/regression_quickstart/report.html to explore the results.")
