import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from cpm import CPMRegression
from cpm.edge_selection import PThreshold, UnivariateEdgeSelection


X = np.load('/spm-data/vault-data3/mmll/projects/cpm_macs_example/datahub/AnalysisReady/hc/DTI_fractional_anisotropy/X.npy')
df = pd.read_csv('/spm-data/vault-data3/mmll/projects/cpm_macs_example/datahub/AnalysisReady/hc/DTI_fractional_anisotropy/subjects.csv',
                 na_values=-99)

X = X[~df['Haushaltsnetto'].isna()]
df = df[~df['Haushaltsnetto'].isna()]
covs = df[['Alter', 'Geschlecht', 'Site']].to_numpy()
y = df['Haushaltsnetto'].to_numpy()

# define edge selection
p_threshold = PThreshold(threshold=[0.001], correction=[None])
p_fdr = PThreshold(threshold=[0.05], correction=['fdr_by'])
univariate_edge_selection = UnivariateEdgeSelection(edge_statistic=['pearson_partial'], edge_selection=[p_threshold])

# run cpm regression
cpm = CPMRegression(results_directory='/spm-data/vault-data3/mmll/projects/cpm_macs_example/haushaltsnetto',
                    cv=KFold(n_splits=10, shuffle=True, random_state=42),
                    edge_selection=univariate_edge_selection,
                    cv_edge_selection=KFold(n_splits=10, shuffle=True, random_state=42),
                    add_edge_filter=True,
                    n_permutations=1000)
results = cpm.estimate(X=X, y=y, covariates=covs)
