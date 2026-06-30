"""
Quickstart: Connectome-based Predictive Modeling for a *binary classification*
target.

This mirrors ``regression_quickstart.py`` but predicts a 0/1 class label instead
of a continuous score. The only real differences are:

  * the target ``y`` is binary,
  * an edge statistic suited to a binary target ('point_biserial'),
  * a stratified cross-validation splitter, and
  * classification metrics in the report (accuracy, balanced accuracy, F1, AUC).

Run it with:

    poetry run python examples/classification_quickstart.py
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold

from cccpm import CPMAnalysis, UnivariateEdgeSelection, PThreshold
from cccpm.simulation.simulate_simple import simulate_confounded_binary_data_chyzhyk

# ---------------------------------------------------------------------------
# 1. Get some data
# ---------------------------------------------------------------------------
# X          : connectome edges, shape (n_samples, n_features)
# y          : the binary outcome (0/1), shape (n_samples,)
# covariates : nuisance variables to control for, shape (n_samples, n_covariates)
#
# We simulate a binary target that is linked to the connectome but also to a
# confound, so confound control matters here too.
X, y, covariates = simulate_confounded_binary_data_chyzhyk(
    n_samples=100, n_features=435, link_type="direct_link"
)

# ---------------------------------------------------------------------------
# 2. Configure edge selection
# ---------------------------------------------------------------------------
# 'point_biserial' is the correlation between each (continuous) edge and the
# binary target. Use 'point_biserial_partial' to control for covariates during
# edge selection.
edge_selection = UnivariateEdgeSelection(
    edge_statistic="point_biserial",
    edge_selection=[PThreshold(threshold=[0.05], correction=[None])],
)

# ---------------------------------------------------------------------------
# 3. Configure and run the analysis
# ---------------------------------------------------------------------------
cpm = CPMAnalysis(
    results_directory="./results/classification_quickstart",
    task_type="classification",                   # or leave as None to auto-detect
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
    edge_selection=edge_selection,
    n_permutations=100,                           # use 1000+ for a real analysis
    device="cpu",                                 # "cuda" uses the GPU if available
)

cpm.run(X=X, y=y, covariates=covariates)

# ---------------------------------------------------------------------------
# 4. Inspect the results
# ---------------------------------------------------------------------------
# Everything is written under the results_directory, including:
#   - cv_results_summary.csv : metrics (accuracy, balanced accuracy, F1, ROC AUC)
#   - p_values.csv           : permutation-based significance
#   - cv_predictions.csv     : out-of-sample predictions per subject
#   - report.html            : a full, human-readable HTML report
print("Done. Open ./results/classification_quickstart/report.html to explore the results.")
