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

from sklearn.model_selection import KFold, ShuffleSplit

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
    n_features=4950,     # a 100-node connectome (100*99/2 = 4950 edges),
                         # matching the built-in Schaefer100 atlas used below
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
# keeping those below a p-value threshold.
#
# Confound control is two independent choices. This one, `selection_input`,
# decides whether the *selection* controls for the covariates: 'residualized'
# tests each edge with the regression y ~ 1 + Z + edge instead of a plain
# correlation. The other, `model_input`, is set on CPMAnalysis below.
#
# Passing several thresholds turns the p-threshold into a tuned hyperparameter;
# the inner CV below picks one per outer fold, so the choice never sees the
# outer test set.
edge_selection = UnivariateEdgeSelection(
    selection_statistic="pearson",
    selection_input="residualized",               # covariates controlled during selection
    edge_selection=[PThreshold(threshold=[0.05, 0.01, 0.001], correction=[None])],
)

# ---------------------------------------------------------------------------
# 3. Configure and run the analysis
# ---------------------------------------------------------------------------
cpm = CPMAnalysis(
    results_directory="./results/regression_quickstart",
    task_type="regression",                       # or leave as None to auto-detect
    cv=KFold(n_splits=10, shuffle=True, random_state=42),
    inner_cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
                                                  # tunes the p-threshold inside each
                                                  # outer fold (nested CV)
    edge_selection=edge_selection,
    model_input="residualized",                   # the second confound-control axis:
                                                  # deconfound the connectome the models
                                                  # consume, after edge selection
    n_permutations=1000,                           # use 1000+ for a real analysis
    atlas="Schaefer100-17",                       # built-in atlas → brain plots in the
                                                  # report; or pass a path to a custom
                                                  # CSV (region,x,y,z[,network]).
                                                  # See cccpm.atlases.list_atlases().
    device="cpu",                                 # "cuda" uses the GPU if available
    stability_significance_method="nbs",          # how *stability* significance is
                                                  # tested -- unrelated to the
                                                  # PThreshold above
    nbs_stability_threshold=0.5,                  # a fraction of folds, not a p-value
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
# This run has confound control on both axes. To see the confound inflation the
# SEM simulator built in on purpose, run it again with selection_input='raw' and
# model_input='raw' and compare the 'connectome' model across the two: the gap is
# the inflation. Within a single run, 'increment' (full - covariates) answers the
# same question -- read it on explained variance, not Pearson r, where a
# difference of correlations is not a statistic and the cell is NaN by design.
print("Done. Open ./results/regression_quickstart/report.html to explore the results.")
