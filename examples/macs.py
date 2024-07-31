import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from cpm import CPMAnalysis


X = np.load('/spm-data/vault-data3/mmll/projects/macs_datahub_example/AnalysisReady/all/FunctionalConnectome/X.npy')
df = pd.read_csv('/spm-data/vault-data3/mmll/projects/macs_datahub_example/AnalysisReady/all/'
                 'FunctionalConnectome/sample.csv',
                 na_values=-99)
X = X[~df['CTQ_Sum'].isna()]
df = df[~df['CTQ_Sum'].isna()]

#X = X[~df['BDI_Sum'].isna()]
#df = df[~df['BDI_Sum'].isna()]
#covs = df[['Alter', 'Geschlecht']].to_numpy()
covs = df[['Geschlecht']].to_numpy()
#y = df['BDI_Sum'].to_numpy()
y = df['CTQ_Sum'].to_numpy()
#covs = df[['Geschlecht']].to_numpy()
y = df['Alter'].to_numpy()


from cpm.edge_selection import PThreshold, SelectPercentile, SelectKBest, UnivariateEdgeSelection
p_threshold = PThreshold(threshold=[0.05], correction=[None])
select_percentile = SelectPercentile(percentile=[0.5])
select_kbest = SelectKBest(k=[5])
univariate_edge_selection = UnivariateEdgeSelection(edge_statistic=['pearson'],
                                                    edge_selection=[p_threshold])

cpm = CPMAnalysis(results_directory='./tmp/macs_demo_age',
                  cv=KFold(n_splits=5, shuffle=True, random_state=42),
                  edge_selection=univariate_edge_selection,
                  cv_edge_selection=KFold(n_splits=2, shuffle=True, random_state=42),
                  estimate_model_increments=True,
                  add_edge_filter=True)
results = cpm.fit(X=X, y=y, covariates=covs)
#print(results)
cpm.permutation_test(X=X, y=y, covariates=covs, n_perms=100)

