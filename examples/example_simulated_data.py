import numpy as np
from sklearn.model_selection import ShuffleSplit, RepeatedKFold

from cccpm import CPMRegression
from cccpm.more_models import RandomForestCPMModel, DecisionTreeCPMModel
from simulation.simulate_simple import simulate_confounded_data_chyzhyk
from cccpm.edge_selection import PThreshold, UnivariateEdgeSelection


X, y, covariates = simulate_confounded_data_chyzhyk(n_samples=1000, n_features=105, link_type='direct_link')
covariates = np.stack([covariates, covariates], axis=1)
univariate_edge_selection = UnivariateEdgeSelection(edge_statistic='pearson',
                                                    edge_selection=[PThreshold(threshold=[0.05, 0.01],
                                                                               correction=[None])],
                                                    t_test_filter=False)

cpm = CPMRegression(results_directory='./tmp/example_simulated_data',
                    cpm_model=DecisionTreeCPMModel,
                    cv=RepeatedKFold(n_splits=2, n_repeats=1, random_state=42),
                    edge_selection=univariate_edge_selection,
                    inner_cv=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
                    n_permutations=2,
                    #atlas_labels='atlas_labels.csv',
                    atlas_labels=None,
                    select_stable_edges=False)

cpm.run(X=X, y=y, covariates=covariates)
