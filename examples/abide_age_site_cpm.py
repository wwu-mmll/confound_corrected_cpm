"""ABIDE I: scan site as a confound for age prediction.

ABIDE pools ~20 scan sites with very different recruited age ranges, and
scanner/sequence differences show up directly in connectivity. Site is thus a
candidate "real" confound -- strongly associated with both the target (age) and
the connectome, unlike sex/motion in `abide_confound_cpm.py`, where correction
costs almost nothing. Runs two cccpm analyses: naive (age ~ connectome) vs.
corrected (age ~ connectome | site); the corrected run adds a "confounds only"
model.
"""
import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
from nilearn.datasets import fetch_abide_pcp
from sklearn.model_selection import RepeatedKFold
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

y = pheno["AGE_AT_SCAN"].astype(float).values
confounds = pd.get_dummies(pheno["SITE_ID"], prefix="site", drop_first=True, dtype=float)
confounds = np.ascontiguousarray(confounds.values)

site_age = pheno.groupby("SITE_ID")["AGE_AT_SCAN"].mean().astype(float)
print(f"n = {len(y)}, age {y.min():.1f}-{y.max():.1f}y, {pheno['SITE_ID'].nunique()} sites")
print(f"Site means (age): {site_age.min():.1f}-{site_age.max():.1f}y "
      f"(SD of site means: {site_age.std():.2f})")


def run(name, covariates, corrected):
    """Run one cccpm analysis and load its summary results."""
    cpm = CPMAnalysis(
        results_directory=f"results_abide_age_site/{name}",
        task_type="regression",
        cv=RepeatedKFold(n_splits=5, n_repeats=5, random_state=42),
        edge_selection=UnivariateEdgeSelection(
            selection_statistic="pearson",
            selection_input="residualized" if corrected else "raw",
            edge_selection=[PThreshold(threshold=[0.01], correction=[None])]),
        model_input="residualized" if corrected else "raw",
        n_permutations=N_PERMUTATIONS, device=DEVICE)
    cpm.run(X, y, covariates)
    return pd.read_csv(f"results_abide_age_site/{name}/cv_results_summary.csv",
                       header=[0, 1], index_col=[0, 1, 2])


naive = run("naive", None, corrected=False)
corrected = run("corrected", confounds, corrected=True)

print("\nr (predicted vs. real age, mean across CV folds):")
print(f"  Connectome, naive          {naive.loc[('connectome', 'both', 0), ('pearson_score', 'mean')]:.3f}")
print(f"  Confounds only (site)      {corrected.loc[('covariates', 'both', 0), ('pearson_score', 'mean')]:.3f}")
print(f"  Connectome, deconfounded   {corrected.loc[('connectome', 'both', 0), ('pearson_score', 'mean')]:.3f}")
print(f"  Confounds + connectome     {corrected.loc[('full', 'both', 0), ('pearson_score', 'mean')]:.3f}")
print("p-values and figures: results_abide_age_site/*/p_values.csv and report.html")
