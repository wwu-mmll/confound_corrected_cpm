import numpy as np
from sklearn.model_selection import ShuffleSplit, RepeatedKFold

from cpm import CPMRegression
from simulation.simulate_data_chyzhyk import simulate_confounded_data_chyzhyk
from cpm.edge_selection import PThreshold, UnivariateEdgeSelection


X, y, covariates = simulate_confounded_data_chyzhyk(n_samples=1000, n_features=4950, link_type='direct_link')
covariates = np.stack([covariates, covariates], axis=1)
univariate_edge_selection = UnivariateEdgeSelection(edge_statistic='pearson',
                                                    edge_selection=[PThreshold(threshold=[0.05, 0.01],
                                                                               correction=[None])],
                                                    t_test_filter=False)

cpm = CPMRegression(results_directory='./tmp/example_simulated_data',
                    cv=RepeatedKFold(n_splits=10, n_repeats=10, random_state=42),
                    edge_selection=univariate_edge_selection,
                    inner_cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=42),
                    n_permutations=1,
                    #atlas_labels='atlas_labels.csv',
                    atlas_labels=None,
                    select_stable_edges=False)

cpm.run(X=X, y=y, covariates=covariates)
#cpm.generate_html_report()
