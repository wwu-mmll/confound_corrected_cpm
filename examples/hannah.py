import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from cpm import CPMRegression
import scipy.io

from cpm.edge_selection import PThreshold, UnivariateEdgeSelection
from cpm.reporting.plots.utils import matrix_to_vector_3d


# Load the .mat file
covariates_file = '/spm-data/vault-data3/TIP-Studie/09_Connectome/0_Projekte/FürNils/design_matrix_SAD_age_sex_med_01000.mat'
covariates = scipy.io.loadmat(covariates_file)['design_matrix_SAD_age_sex_med_01000']
y = covariates[:, 1]
covs = covariates[:, 2:]

brain_file = '/spm-data/vault-data3/TIP-Studie/09_Connectome/0_Projekte/FürNils/Connectivity_dti_nos_Aparc_stacked.mat'
connectome = scipy.io.loadmat(brain_file)['connectivityAparcstacked']
connectome = np.moveaxis(connectome, -1, 0)
X = matrix_to_vector_3d(connectome)

atlas_file = "/spm-data/vault-data3/TIP-Studie/09_Connectome/0_Projekte/FürNils/brainRegions_aparc.csv"

p_threshold = PThreshold(threshold=[0.01], correction=[None])
univariate_edge_selection = UnivariateEdgeSelection(edge_statistic=['pearson'],
                                                    edge_selection=[p_threshold])

cpm_regression = CPMRegression(results_directory='./tmp/hannah_sad_pearson',
                               cv=KFold(n_splits=10, shuffle=True, random_state=42),
                               edge_selection=univariate_edge_selection,
                               #cv_edge_selection=KFold(n_splits=2, shuffle=True, random_state=42),
                               add_edge_filter=True,
                               n_permutations=1000,
                               atlas_labels=atlas_file)
results = cpm_regression.estimate(X=X, y=y, covariates=covs)
#cpm_regression._calculate_permutation_results()
