import numpy as np
from sklearn.model_selection import ShuffleSplit, RepeatedStratifiedKFold

from cccpm import CPMAnalysis
from cccpm.simulation.simulate_simple import simulate_confounded_binary_data_chyzhyk
from cccpm.edge_selection import PThreshold, UnivariateEdgeSelection


X, y, covariates = simulate_confounded_binary_data_chyzhyk(n_samples=100, n_features=1225, link_type='direct_link')
univariate_edge_selection = UnivariateEdgeSelection(edge_statistic='point_biserial',
                                                    edge_selection=[PThreshold(threshold=[0.05, 0.01],
                                                                               correction=[None])],
                                                    t_test_filter=False)

cpm = CPMAnalysis(results_directory='./tmp/example_simulated_classification',
                  task_type='classification',
                  cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=42),
                  edge_selection=univariate_edge_selection,
                  inner_cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
                  n_permutations=1000,
                  select_stable_edges=False,
                  device='cuda')

cpm.run(X=X, y=y, covariates=covariates)
