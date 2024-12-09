import os
import pandas as pd
import numpy as np
from cpm.reporting.plots.utils import matrix_to_vector_3d
from sklearn.model_selection import RepeatedKFold, KFold

from cpm import CPMRegression
from cpm.edge_selection import PThreshold, UnivariateEdgeSelection


data_folder = "/spm-data/vault-data3/mmll/data/HCP/100_nodes_times_series/"
X_file = "functional_connectivity.npy"
ids_file = "subject_ids.csv"
meta_file = "RESTRICTED_NilsWinter_11_26_2024_8_3_14.csv"
unres_meta_file = "unrestricted_NilsWinter_12_4_2024_10_11_45.csv"

X = np.load(os.path.join(data_folder, X_file))
ids = pd.read_csv(os.path.join(data_folder, ids_file))
ids['Subject'] = ids['Subject_ID']
df = pd.read_csv(os.path.join(data_folder, meta_file))
df_unres = pd.read_csv(os.path.join(data_folder, unres_meta_file))

df = pd.merge(ids, df, how="left", on="Subject")
df = pd.merge(df, df_unres, how="left", on="Subject")

df['Sex'] = pd.get_dummies(df['Gender'], drop_first=True, dtype='int').to_numpy()

X = matrix_to_vector_3d(X)
#target = "BMI"
target = "SSAGA_Income"
#target = "Age_in_Yrs"

X = X[~df[target].isna()]
df = df[~df[target].isna()]

#X = X[~df['BMI'].isna()]
#df = df[~df['BMI'].isna()]

#edge_statistics = ['pearson', 'pearson_partial']
edge_statistics = ['spearman']

#covariates = df['BMI'].to_numpy().reshape(-1, 1)
covariates = df[['Age_in_Yrs', 'Sex']].to_numpy()


for edge_statistic in edge_statistics:
    univariate_edge_selection = UnivariateEdgeSelection(edge_statistic=[edge_statistic],
                                                        edge_selection=[
                                                            PThreshold(threshold=[0.1, 0.05, 0.01, 0.005],
                                                                                  correction=[None]),
                                                            #            PThreshold(threshold=[0.05],
                                                            #                       correction=['fdr_by'])
                                                                                   ])
    cpm = CPMRegression(results_directory=f'{data_folder}/results/hcp_{target}_{edge_statistic}',
                        cv=KFold(n_splits=10, shuffle=True, random_state=42),
                        edge_selection=univariate_edge_selection,
                        inner_cv=KFold(n_splits=10, shuffle=True, random_state=42),
                        select_stable_edges=True,
                        stability_threshold=0,
                        n_permutations=20)
    cpm.estimate(X=X, y=df[target].to_numpy(), covariates=covariates)
print()