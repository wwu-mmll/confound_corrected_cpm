"""
ABIDE I: Standort (Site) als Confound bei der Altersvorhersage

ABIDE bündelt ~20 Scan-Standorte mit sehr unterschiedlichen Altersbereichen der
rekrutierten Stichproben, und Scanner-/Sequenzunterschiede schlagen sich direkt in
der Konnektivität nieder. Site ist damit ein Kandidat für einen "echten" Confound:
stark mit dem Target (Alter) UND mit dem Connectome assoziiert -- anders als
Geschlecht/Bewegung in `abide_confound_cpm.py`, wo die Korrektur praktisch nichts
kostet. Das Skript rechnet zwei cccpm-Analysen:
  1. naiv:        Alter ~ Connectome
  2. korrigiert:  Alter ~ Connectome | Site
Die korrigierte Analyse liefert zusätzlich das Modell "nur Confounds".
"""
import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
from nilearn.datasets import fetch_abide_pcp
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

# --- Daten laden (ROI-Zeitreihen, CC200-Atlas, ~870 Probanden)
data = fetch_abide_pcp(derivatives="rois_cc200", pipeline="cpac",
                       band_pass_filtering=True, quality_checked=True)
pheno = pd.DataFrame(data.phenotypic)

# --- Connectomes: Korrelationsmatrizen (n_subjects x 200 x 200) -> Edge-Vektor
X = ConnectivityMeasure(kind="correlation").fit_transform(data.rois_cc200)
X = np.nan_to_num(X)  # leere ROIs einzelner Probanden
X = X[:, *np.triu_indices(X.shape[1], k=1)]  # oberes Dreieck -> n_subjects x n_edges

y = pheno["AGE_AT_SCAN"].astype(float).values
confounds = pd.get_dummies(pheno["SITE_ID"], prefix="site", drop_first=True, dtype=float)
confounds = np.ascontiguousarray(confounds.values)

site_age = pheno.groupby("SITE_ID")["AGE_AT_SCAN"].mean().astype(float)
print(f"n = {len(y)}, Alter {y.min():.1f}-{y.max():.1f} J., {pheno['SITE_ID'].nunique()} Standorte")
print(f"Standortmittel (Alter): {site_age.min():.1f}-{site_age.max():.1f} J. "
      f"(Standardabweichung der Standortmittel: {site_age.std():.2f})")


def run(name, covariates, corrected):
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

print("\nr (vorhergesagtes vs. echtes Alter, Mittel über CV-Folds):")
print(f"  Connectome, naiv           {naive.loc[('connectome', 'both', 0), ('pearson_score', 'mean')]:.3f}")
print(f"  nur Confounds (Site)       {corrected.loc[('covariates', 'both', 0), ('pearson_score', 'mean')]:.3f}")
print(f"  Connectome, deconfounded   {corrected.loc[('connectome', 'both', 0), ('pearson_score', 'mean')]:.3f}")
print(f"  Confounds + Connectome     {corrected.loc[('full', 'both', 0), ('pearson_score', 'mean')]:.3f}")
print("p-Werte und Abbildungen: results_abide_age_site/*/p_values.csv und report.html")
