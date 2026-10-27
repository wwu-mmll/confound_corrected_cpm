"""
Development fMRI (Richardson et al. 2018): Kopfbewegung als Confound bei der Altersvorhersage

122 Kinder (3-12 J.) schauen einen Kurzfilm. Jüngere Kinder bewegen sich mehr, und
Kopfbewegung verändert die Konnektivität. Das Skript rechnet zwei cccpm-Analysen:
  1. naiv:        Alter ~ Connectome
  2. korrigiert:  Alter ~ Connectome | Kopfbewegung, Geschlecht
Die korrigierte Analyse liefert zusätzlich das Modell "nur Confounds".
Hier gibt es auch einen echten Alterseffekt: die Vorhersage schrumpft, statt zu verschwinden.
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

# GPU: MAGMA im torch-cu130-Wheel hat keinen Code für Turing (sm_75) -> cuSOLVER nutzen
torch.backends.cuda.preferred_linalg_library("cusolver")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Der Covariate-Pairplot von cccpm dauert bei ~20 Site-Dummies sehr lange -> abschalten
_di.pairplot_flexible = lambda df, path: None

N_PERMUTATIONS = 100  # fürs Paper 1000

# --- Daten laden (vorverarbeitet mit fMRIPrep)
data = fetch_development_fmri(age_group="child")
pheno = pd.DataFrame(data.phenotypic)
masker = NiftiMapsMasker(maps_img=fetch_atlas_msdl().maps, standardize="zscore_sample",
                         detrend=True, low_pass=0.1, high_pass=0.01, t_r=2.0)

# --- Zeitreihen extrahieren (Nuisance-Regression) und mittlere Kopfbewegung berechnen
timeseries, mean_fd = [], []
for func, confound_file in zip(data.func, data.confounds):
    conf = pd.read_csv(confound_file, sep="\t")
    mean_fd.append(conf["framewise_displacement"].mean())
    nuisance = conf.drop(columns="framewise_displacement").fillna(0).values
    timeseries.append(masker.fit_transform(func, confounds=nuisance))

# --- Connectomes: Korrelationsmatrizen (n_subjects x 39 x 39) -> Edge-Vektor
X = ConnectivityMeasure(kind="correlation").fit_transform(timeseries)
X = X[:, *np.triu_indices(X.shape[1], k=1)]  # oberes Dreieck -> n_subjects x n_edges

y = pheno["Age"].astype(float).values
confounds = np.ascontiguousarray(np.column_stack([
    mean_fd,                                          # Kopfbewegung
    (pheno["Gender"] == "M").astype(float),           # Geschlecht
]))

print(f"n = {len(y)}, Alter {y.min():.1f}-{y.max():.1f} J.")
print(f"Korrelation Alter vs. Kopfbewegung: r = {np.corrcoef(y, mean_fd)[0, 1]:.2f}")


def run(name, covariates, corrected):
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

print("\nr (vorhergesagtes vs. echtes Alter, Mittel über CV-Folds):")
print(f"  Connectome, naiv           {naive.loc[('connectome', 'both', 0), ('pearson_score', 'mean')]:.3f}")
print(f"  nur Confounds              {corrected.loc[('covariates', 'both', 0), ('pearson_score', 'mean')]:.3f}")
print(f"  Connectome, deconfounded   {corrected.loc[('connectome', 'both', 0), ('pearson_score', 'mean')]:.3f}")
print(f"  Confounds + Connectome     {corrected.loc[('full', 'both', 0), ('pearson_score', 'mean')]:.3f}")
print("p-Werte und Abbildungen: results_development/*/p_values.csv und report.html")