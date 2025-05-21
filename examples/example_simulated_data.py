from sklearn.model_selection import KFold, ShuffleSplit, RepeatedKFold

from cpm import CPMRegression
from cpm.simulate_data import simulate_regression_data
from cpm.edge_selection import PThreshold, UnivariateEdgeSelection


X, y, covariates = simulate_regression_data(n_features=1225, n_informative_features=50,
                                            covariate_effect_size=0.2,
                                            feature_effect_size=100,
                                            noise_level=0.1)

univariate_edge_selection = UnivariateEdgeSelection(edge_statistic='pearson',
                                                    edge_selection=[PThreshold(threshold=[0.05, 0.01],
                                                                               correction=[None])],
                                                    t_test_filter=False)

cpm = CPMRegression(results_directory='./tmp/example_simulated_data',
                    cv=RepeatedKFold(n_splits=10, n_repeats=1, random_state=42),
                    edge_selection=univariate_edge_selection,
                    inner_cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
                    n_permutations=100,
                    atlas_labels='atlas_labels.csv',
                    #atlas_labels=None,
                    select_stable_edges=False)

cpm.run(X=X, y=y, covariates=covariates)
#cpm.generate_html_report()
