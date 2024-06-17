import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from cpm import CPMAnalysis


X = np.load('/spm-data/vault-data3/mmll/projects/macs_datahub_example/AnalysisReady/all/FunctionalConnectome/X.npy')
df = pd.read_csv('/spm-data/vault-data3/mmll/projects/macs_datahub_example/AnalysisReady/all/'
                 'FunctionalConnectome/sample.csv',
                 na_values=-99)
X = X[~df['BDI_Sum'].isna()]
df = df[~df['BDI_Sum'].isna()]
covs = df[['Alter', 'Geschlecht']].to_numpy()
y = df['BDI_Sum'].to_numpy()
#covs = df[['Geschlecht']].to_numpy()
#y = df['Alter'].to_numpy()

cpm = CPMAnalysis(results_directory='./tmp/macs_demo',
                  cv=KFold(n_splits=10, shuffle=True, random_state=42))
results = cpm.fit(X=X, y=y, covariates=covs)
print(results)
p_pos, p_neg = cpm.permutation_test(X=X, y=y, covariates=covs, n_perms=30)

print(p_pos)
print(p_neg)
