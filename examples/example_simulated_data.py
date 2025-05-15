from sklearn.model_selection import KFold, ShuffleSplit

from cpm import CPMRegression
from cpm.simulate_data import simulate_regression_data
from cpm.edge_selection import PThreshold, UnivariateEdgeSelection


X, y, covariates = simulate_regression_data(n_features=1225, n_informative_features=50,
                                            covariate_effect_size=0.2,
                                            feature_effect_size=100,
                                            noise_level=0.1)

univariate_edge_selection = UnivariateEdgeSelection(edge_statistic='pearson',
                                                    edge_selection=[PThreshold(threshold=[0.05],
                                                                               correction=[None])],
                                                    t_test_filter=False)

cpm = CPMRegression(results_directory='./tmp/example_simulated_data',
                    cv=KFold(n_splits=5, shuffle=True, random_state=42),
                    edge_selection=univariate_edge_selection,
                    inner_cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
                    n_permutations=2,
                    atlas_labels='atlas_labels.csv')

cpm.run(X=X, y=y, covariates=covariates)
