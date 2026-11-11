"""ABIDE I: confound effect in CPM (autism vs. control).

ABIDE is strongly confounded: mostly male, ~20 sites, and autistic subjects move
more in the scanner. Runs two cccpm analyses: naive (autism ~ connectome) vs.
corrected (autism ~ connectome | sex, age, site, motion); the corrected run adds
a "confounds only" model.
"""
import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
from nilearn.datasets import fetch_abide_pcp
from sklearn.model_selection import RepeatedStratifiedKFold
from cccpm import CPMAnalysis, UnivariateEdgeSelection, PThreshold
import torch
import cccpm.reporting.data_insights as _di

# MAGMA in the torch-cu130 wheel has no Turing (sm_75) kernels -> use cuSOLVER
torch.backends.cuda.preferred_linalg_library("cusolver")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# cccpm's covariate pairplot is slow with ~20 site dummies -> disable it
_di.pairplot_flexible = lambda df, path: None

N_PERMUTATIONS = 100  # use 1000 for the paper

data = fetch_abide_pcp(derivatives="rois_cc200", pipeline="cpac",
                       band_pass_filtering=True, quality_checked=True)
pheno = pd.DataFrame(data.phenotypic)

X = ConnectivityMeasure(kind="correlation").fit_transform(data.rois_cc200)
X = np.nan_to_num(X)  # a few subjects have empty ROIs
X = X[:, *np.triu_indices(X.shape[1], k=1)]

y = (pheno["DX_GROUP"] == 1).astype(float).values           # 1 = autism, 2 = control
confounds = pd.concat([
    (pheno["SEX"] == 1).astype(float).rename("male"),        # 1 = male
    pheno["AGE_AT_SCAN"].astype(float),
    pheno["func_mean_fd"].astype(float),                     # head motion
    pd.get_dummies(pheno["SITE_ID"], prefix="site", drop_first=True, dtype=float),
], axis=1)
confounds = np.ascontiguousarray(confounds.values)

print(f"n = {len(y)}, autism = {int(y.sum())}")
print(f"Proportion male: autism {confounds[y == 1, 0].mean():.0%}, "
      f"control {confounds[y == 0, 0].mean():.0%}")
print(f"Mean FD:         autism {confounds[y == 1, 2].mean():.3f}, "
      f"control {confounds[y == 0, 2].mean():.3f}")


def run(name, covariates, corrected):
    """Run one cccpm analysis and load its summary results."""
    cpm = CPMAnalysis(
        results_directory=f"results_abide/{name}",
        task_type="classification",
        cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42),
        edge_selection=UnivariateEdgeSelection(
            selection_statistic="pearson",
            selection_input="residualized" if corrected else "raw",
            edge_selection=[PThreshold(threshold=[0.01], correction=[None])]),
        model_input="residualized" if corrected else "raw",
        n_permutations=N_PERMUTATIONS, device=DEVICE)
    cpm.run(X, y, covariates)
    return pd.read_csv(f"results_abide/{name}/cv_results_summary.csv",
                       header=[0, 1], index_col=[0, 1, 2])


naive = run("naive", None, corrected=False)
corrected = run("corrected", confounds, corrected=True)

print("\nROC-AUC (mean across CV folds):")
print(f"  Connectome, naive          {naive.loc[('connectome', 'both', 0), ('roc_auc', 'mean')]:.3f}")
print(f"  Confounds only             {corrected.loc[('covariates', 'both', 0), ('roc_auc', 'mean')]:.3f}")
print(f"  Connectome, deconfounded   {corrected.loc[('connectome', 'both', 0), ('roc_auc', 'mean')]:.3f}")
print(f"  Confounds + connectome     {corrected.loc[('full', 'both', 0), ('roc_auc', 'mean')]:.3f}")
print("p-values and figures: results_abide/*/p_values.csv and report.html")