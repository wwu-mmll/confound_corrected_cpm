"""Quickstart: CPM for a binary classification target (see regression_quickstart.py).

Run it with:

    poetry run python examples/classification_quickstart.py
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold

from cccpm import CPMAnalysis, UnivariateEdgeSelection, PThreshold
from cccpm.simulation.simulate_sem import simulate_data_given_kappa

# Binarise the continuous SEM outcome at its median; the confound is still baked
# into the connectome, so confound control matters here too.
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
X, covariates = sim["X"], sim["Z"]
y = (sim["y"].ravel() > np.median(sim["y"])).astype(float)

# A binary target through the Pearson path *is* the point-biserial correlation,
# so there is no separate statistic to pick.
edge_selection = UnivariateEdgeSelection(
    selection_statistic="pearson",
    selection_input="residualized",
    edge_selection=[PThreshold(threshold=[0.05], correction=[None])],
)

cpm = CPMAnalysis(
    results_directory="./results/classification_quickstart",
    task_type="classification",
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
    edge_selection=edge_selection,
    n_permutations=1000,
    atlas="Schaefer100-17",
    device="cpu",
)

cpm.run(X=X, y=y, covariates=covariates)

print("Done. Open ./results/classification_quickstart/report.html to explore the results.")
