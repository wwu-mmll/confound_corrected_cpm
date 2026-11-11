"""Development fMRI (Richardson et al. 2018): head motion as a confound for age prediction.

122 children (3-12y) watch a short film. Younger children move more, and head
motion changes connectivity. Runs two cccpm analyses: naive (age ~ connectome)
vs. corrected (age ~ connectome | motion, sex); the corrected run adds a
"confounds only" model. Here there's also a real age effect: the prediction
shrinks rather than vanishing.
"""
import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
from nilearn.datasets import fetch_atlas_msdl, fetch_development_fmri
from nilearn.maskers import NiftiMapsMasker
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

data = fetch_development_fmri(age_group="child")
pheno = pd.DataFrame(data.phenotypic)
masker = NiftiMapsMasker(maps_img=fetch_atlas_msdl().maps, standardize="zscore_sample",
                         detrend=True, low_pass=0.1, high_pass=0.01, t_r=2.0)

timeseries, mean_fd = [], []
for func, confound_file in zip(data.func, data.confounds):
    conf = pd.read_csv(confound_file, sep="\t")
    mean_fd.append(conf["framewise_displacement"].mean())
    nuisance = conf.drop(columns="framewise_displacement").fillna(0).values
    timeseries.append(masker.fit_transform(func, confounds=nuisance))

X = ConnectivityMeasure(kind="correlation").fit_transform(timeseries)
X = X[:, *np.triu_indices(X.shape[1], k=1)]

y = pheno["Age"].astype(float).values
confounds = np.ascontiguousarray(np.column_stack([
    mean_fd,                                          # head motion
    (pheno["Gender"] == "M").astype(float),           # sex
]))

print(f"n = {len(y)}, age {y.min():.1f}-{y.max():.1f}y")
print(f"Correlation age vs. head motion: r = {np.corrcoef(y, mean_fd)[0, 1]:.2f}")


def run(name, covariates, corrected):
    """Run one cccpm analysis and load its summary results."""
    cpm = CPMAnalysis(
        results_directory=f"results_development/{name}",
        task_type="regression",
        cv=RepeatedKFold(n_splits=5, n_repeats=5, random_state=42),
        edge_selection=UnivariateEdgeSelection(
            selection_statistic="pearson",
            selection_input="residualized" if corrected else "raw",
            edge_selection=[PThreshold(threshold=[0.01], correction=[None])]),
        model_input="residualized" if corrected else "raw",
        n_permutations=N_PERMUTATIONS, device=DEVICE)
    cpm.run(X, y, covariates)
    return pd.read_csv(f"results_development/{name}/cv_results_summary.csv",
                       header=[0, 1], index_col=[0, 1, 2])


naive = run("naive", None, corrected=False)
corrected = run("corrected", confounds, corrected=True)

print("\nr (predicted vs. real age, mean across CV folds):")
print(f"  Connectome, naive          {naive.loc[('connectome', 'both', 0), ('pearson_score', 'mean')]:.3f}")
print(f"  Confounds only             {corrected.loc[('covariates', 'both', 0), ('pearson_score', 'mean')]:.3f}")
print(f"  Connectome, deconfounded   {corrected.loc[('connectome', 'both', 0), ('pearson_score', 'mean')]:.3f}")
print(f"  Confounds + connectome     {corrected.loc[('full', 'both', 0), ('pearson_score', 'mean')]:.3f}")
print("p-values and figures: results_development/*/p_values.csv and report.html")