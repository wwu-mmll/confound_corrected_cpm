from sklearn.model_selection import RepeatedKFold

from cpm import CPMRegression
from cpm.simulation.simulate_data import simulate_regression_data_scenarios
from cpm.edge_selection import PThreshold, UnivariateEdgeSelection


link_types = ['no_link',
              'no_no_link',
              'direct_link',
              'weak_link'
            ]
edge_statistics = ['pearson', 'pearson_partial']
results_folder = '/spm-data/vault-data3/mmll/projects/confound_corrected_cpm/results'

for link in link_types:
    for edge_statistic in edge_statistics:
        X, y, covariates = simulate_regression_data_scenarios(n_features=1225, n_informative_features=50, link_type=link)

        univariate_edge_selection = UnivariateEdgeSelection(edge_statistic=[edge_statistic],
                                                            edge_selection=[PThreshold(threshold=[0.05],
                                                                                       correction=['fdr_by'])])
        cpm = CPMRegression(results_directory=f'{results_folder}/simulated_data_{link}_{edge_statistic}',
                            cv=RepeatedKFold(n_splits=10, n_repeats=5, random_state=42),
                            edge_selection=univariate_edge_selection,
                            #cv_edge_selection=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
                            add_edge_filter=True,
                            n_permutations=2)
        cpm.run(X=X, y=y, covariates=covariates)

        #cpm._calculate_permutation_results('./tmp/example_simulated_data2')

