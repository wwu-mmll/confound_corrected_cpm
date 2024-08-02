from sklearn.model_selection import KFold, ShuffleSplit

from cpm import CPMRegression
from cpm.simulate_data import simulate_regression_data
from cpm.edge_selection import PThreshold, UnivariateEdgeSelection


X, y, covariates = simulate_regression_data(n_features=1225, n_informative_features=20,
                                            covariate_effect_size=50,
                                            feature_effect_size=0.01,
                                            noise_level=5)

univariate_edge_selection = UnivariateEdgeSelection(edge_statistic=['pearson'],
                                                    edge_selection=[PThreshold(threshold=[0.1],
                                                                               correction=[None])])

cpm = CPMRegression(results_directory='./tmp/example_simulated_data2',
                    cv=KFold(n_splits=5, shuffle=True, random_state=42),
                    edge_selection=univariate_edge_selection,
                    cv_edge_selection=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
                    add_edge_filter=True,
                    n_permutations=100)
cpm.save_configuration(config_filename='./tmp/config.pkl')

cpm_remote = CPMRegression(results_directory='./tmp/example_simulated_data2')
cpm_remote.load_configuration(results_directory='./tmp/example_simulated_data3',
                              config_filename='./tmp/config.pkl')
