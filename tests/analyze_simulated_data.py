from sklearn.model_selection import RepeatedKFold

from cccpm import CPMRegression
from cccpm.simulation.simulate_simple import simulate_confounded_data_chyzhyk
from cccpm.edge_selection import PThreshold, UnivariateEdgeSelection


link_types = ['no_link',
              'direct_link',
              'weak_link'
            ]
edge_statistics = ['pearson', 'pearson_partial']
results_folder = '/spm-data/vault-data3/mmll/projects/confound_corrected_cpm/results'

for link in link_types:
    for edge_statistic in edge_statistics:
        X, y, covariates = simulate_confounded_data_chyzhyk(n_features=1225, link_type=link)

        univariate_edge_selection = UnivariateEdgeSelection(edge_statistic=[edge_statistic],
                                                            edge_selection=[PThreshold(threshold=[0.05],
                                                                                       correction=['fdr_by'])])
        cpm = CPMRegression(results_directory=f'{results_folder}/simulated_data_{link}_{edge_statistic}',
                            cv=RepeatedKFold(n_splits=10, n_repeats=5, random_state=42),
                            edge_selection=univariate_edge_selection,
                            n_permutations=2)
        cpm.run(X=X, y=y, covariates=covariates)


