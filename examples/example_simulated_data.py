from sklearn.model_selection import KFold, ShuffleSplit

from cpm import CPMAnalysis
from cpm.simulate_data import simulate_data
from cpm.edge_selection import PThreshold, UnivariateEdgeSelection


X, y, covariates = simulate_data(n_features=1225, n_informative_features=20,
                                 covariate_effect_size=50,
                                 feature_effect_size=0.01,
                                 noise_level=5)

univariate_edge_selection = UnivariateEdgeSelection(edge_statistic=['pearson'],
                                                    edge_selection=[PThreshold(threshold=[0.1],
                                                                               correction=[None])])

cpm = CPMAnalysis(results_directory='./tmp/example_simulated_data',
                  cv=KFold(n_splits=5, shuffle=True, random_state=42),
                  edge_selection=univariate_edge_selection,
                  cv_edge_selection=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
                  estimate_model_increments=True,
                  add_edge_filter=True)
cpm.fit(X=X, y=y, covariates=covariates)
cpm.permutation_test(X=X, y=y, covariates=covariates, n_perms=100)

