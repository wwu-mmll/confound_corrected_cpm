"""
Generate simulated connectome data with a known amount of confounding and run a
full CPM analysis, including nested CV for the p-threshold and an HTML report.

The data come from the SEM-based simulator (``cccpm.simulation.simulate_sem``),
which builds a connectome/outcome pair from an explicit structural model: a latent
confound Z is a common cause of both, and ``kappa`` sets the fraction of the naive
brain–outcome R² that is confound-driven. This gives simulated data with an
*analytically known* ground truth. See the "Simulating Data" docs page for details.
"""

import numpy as np
from sklearn.model_selection import ShuffleSplit, RepeatedKFold

from cccpm import CPMAnalysis
from cccpm.simulation.simulate_sem import simulate_data_given_kappa
from cccpm.edge_selection import PThreshold, UnivariateEdgeSelection


# A 50-node connectome (50*49/2 = 1225 edges) with moderate confounding.
sim = simulate_data_given_kappa(
    R2_X_y=0.4,
    kappa=0.3,
    n_features=1225,
    n_features_informative=60,     # mixed edges: signal + confound leakage
    n_pure_signal_features=30,     # edges tied to y but not the confound
    n_confound_only_features=30,   # edges tied to y only through the confound
    n_confounds=2,
    n_samples=150,
    random_state=42,
)
X, y, covariates = sim["X"], sim["y"], sim["Z"]

univariate_edge_selection = UnivariateEdgeSelection(edge_statistic='pearson',
                                                    edge_selection=[PThreshold(threshold=[0.05, 0.01],
                                                                               correction=[None])])

cpm = CPMAnalysis(results_directory='./tmp/example_simulated_data',
                  cv=RepeatedKFold(n_splits=10, n_repeats=1, random_state=42),
                  edge_selection=univariate_edge_selection,
                  inner_cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
                  n_permutations=100,
                  #atlas_labels='atlas_labels.csv',
                  select_stable_edges=False,
                  device='cpu')

cpm.run(X=X, y=y, covariates=covariates)
cpm.generate_html_report()
