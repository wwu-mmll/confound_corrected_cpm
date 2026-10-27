"""
ABIDE I: Confound-Effekt in CPM (Autismus vs. Kontrolle)

ABIDE ist stark konfundiert: überwiegend männlich, ~20 Standorte, und Probanden mit
Autismus bewegen sich im Scanner mehr. Das Skript rechnet zwei cccpm-Analysen:
  1. naiv:        Autismus ~ Connectome
  2. korrigiert:  Autismus ~ Connectome | Geschlecht, Alter, Site, Kopfbewegung
Die korrigierte Analyse liefert zusätzlich das Modell "nur Confounds".
"""
import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
from nilearn.datasets import fetch_abide_pcp
from sklearn.model_selection import RepeatedStratifiedKFold
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

y = (pheno["DX_GROUP"] == 1).astype(float).values           # 1 = Autismus, 2 = Kontrolle
confounds = pd.concat([
    (pheno["SEX"] == 1).astype(float).rename("male"),        # 1 = männlich
    pheno["AGE_AT_SCAN"].astype(float),
    pheno["func_mean_fd"].astype(float),                     # Kopfbewegung
    pd.get_dummies(pheno["SITE_ID"], prefix="site", drop_first=True, dtype=float),
], axis=1)
confounds = np.ascontiguousarray(confounds.values)

print(f"n = {len(y)}, Autismus = {int(y.sum())}")
print(f"Anteil männlich: Autismus {confounds[y == 1, 0].mean():.0%}, "
      f"Kontrolle {confounds[y == 0, 0].mean():.0%}")
print(f"Mittlere FD:     Autismus {confounds[y == 1, 2].mean():.3f}, "
      f"Kontrolle {confounds[y == 0, 2].mean():.3f}")


def run(name, covariates, corrected):
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

print("\nROC-AUC (Mittel über CV-Folds):")
print(f"  Connectome, naiv           {naive.loc[('connectome', 'both', 0), ('roc_auc', 'mean')]:.3f}")
print(f"  nur Confounds              {corrected.loc[('covariates', 'both', 0), ('roc_auc', 'mean')]:.3f}")
print(f"  Connectome, deconfounded   {corrected.loc[('connectome', 'both', 0), ('roc_auc', 'mean')]:.3f}")
print(f"  Confounds + Connectome     {corrected.loc[('full', 'both', 0), ('roc_auc', 'mean')]:.3f}")
print("p-Werte und Abbildungen: results_abide/*/p_values.csv und report.html")