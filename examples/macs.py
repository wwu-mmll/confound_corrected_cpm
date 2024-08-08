import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from cpm import CPMRegression


#X = np.load('/spm-data/vault-data3/mmll/projects/macs_datahub_example/AnalysisReady/all/FunctionalConnectome/X.npy')
#df = pd.read_csv('/spm-data/vault-data3/mmll/projects/macs_datahub_example/AnalysisReady/all/'
#                 'FunctionalConnectome/sample.csv',
#                 na_values=-99)
X = np.load('/spm-data/vault-data3/mmll/projects/microbiome_mdd_eggerthella/data/AnalysisReady/sample_leon/DTI_fractional_anisotropy/X.npy')
df = pd.read_csv('/spm-data/vault-data3/mmll/projects/microbiome_mdd_eggerthella/data/AnalysisReady/sample_leon/DTI_fractional_anisotropy/subjects.csv',
                 na_values=-99)

X = X[df['Group'] == 1]
df = df[df['Group'] == 1]


X = X[~df['CTQ_Sum'].isna()]
df = df[~df['CTQ_Sum'].isna()]

#X = X[~df['BDI_Sum'].isna()]
#df = df[~df['BDI_Sum'].isna()]
covs = df[['Alter', 'Geschlecht', 'Site']].to_numpy()
#covs = df[['Geschlecht']].to_numpy()
#y = df['BDI_Sum'].to_numpy()
y = df['CTQ_Sum'].to_numpy()
#covs = df[['Geschlecht']].to_numpy()
#y = df['Alter'].to_numpy()


from cpm.edge_selection import PThreshold, UnivariateEdgeSelection
p_threshold = PThreshold(threshold=[0.05], correction=[None])
univariate_edge_selection = UnivariateEdgeSelection(edge_statistic=['pearson_partial'],
                                                    edge_selection=[p_threshold])

cpm = CPMRegression(results_directory='./tmp/macs_ctq_hc',
                    cv=KFold(n_splits=10, shuffle=True, random_state=42),
                    edge_selection=univariate_edge_selection,
                    #cv_edge_selection=KFold(n_splits=2, shuffle=True, random_state=42),
                    add_edge_filter=True,
                    n_permutations=1000)
results = cpm.estimate(X=X, y=y, covariates=covs)
#print(results)
cpm._calculate_permutation_results()

