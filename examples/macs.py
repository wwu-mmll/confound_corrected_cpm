import numpy as np
import pandas as pd


X = np.load('/spm-data/vault-data3/mmll/projects/macs_datahub_example/AnalysisReady/all/FunctionalConnectome/X.npy')
df = pd.read_csv('/spm-data/vault-data3/mmll/projects/macs_datahub_example/AnalysisReady/all/FunctionalConnectome/sample.csv',
                 na_values=-99)
covs = df[['Alter', 'Geschlecht']]
y = df['BDI_Sum']

print()
