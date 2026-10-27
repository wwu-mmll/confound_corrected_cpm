"""
ABIDE I: Negativkontrolle -- zeigt, dass Confound-Korrektur einen falschen Befund repariert

`abide_age_site_cpm.py` zeigt, dass die Alters-Vorhersage nach Site-Korrektur einbricht
(r 0,564 -> 0,286). Mit echten Daten lässt sich daraus aber nicht beweisen, dass der naive
Befund *falsch* war -- man weiß nie, wie viel des Effekts echtes Altern und wie viel
Scanner-Artefakt ist (siehe `confound_examples_notes.md`: η² = 0,49 Alters-Varianz liegt
zwischen den Standorten).

Dieses Skript konstruiert stattdessen eine Negativkontrolle mit bekannter Ground Truth:
Das Alter wird INNERHALB jedes Standorts zufällig permutiert. Das zerstört jeden
individuellen Zusammenhang zwischen einer Person und ihrem (permutierten) "Alter" --
der wahre Connectome-Alter-Zusammenhang ist jetzt per Konstruktion exakt null. Der
Standortmittelwert des Alters bleibt dabei unverändert (nur innerhalb der Standorte wird
gemischt), also bleibt auch die Site<->Alter-Beziehung exakt erhalten.

Erwartung:
  naiv:        findet trotzdem einen "signifikanten" Connectome-Effekt -- ein Fehlalarm,
               der ausschließlich daher kommt, dass Site sowohl mit dem (permutierten)
               Alter als auch mit dem Connectome zusammenhängt (Scanner-Artefakte).
  korrigiert:  der Effekt verschwindet (keine Signifikanz mehr) -- die Korrektur repariert
               den Fehlalarm korrekt, weil es hier nichts Echtes mehr zu entdecken gibt.

Anders als `abide_age_site_cpm.py` ist das kein Beispiel fürs Paper-Narrativ ("hier ist ein
plausibler Use Case"), sondern ein Validitätsnachweis: ein Fall mit bekannter Ground Truth
(Effekt = 0), an dem sich zeigen lässt, dass die Korrektur tut, was sie soll.
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

N_PERMUTATIONS = 1000
RNG = np.random.default_rng(42)

# --- Daten laden (ROI-Zeitreihen, CC200-Atlas, ~870 Probanden)
data = fetch_abide_pcp(derivatives="rois_cc200", pipeline="cpac",
                       band_pass_filtering=True, quality_checked=True)
pheno = pd.DataFrame(data.phenotypic).reset_index(drop=True)

# --- Connectomes: Korrelationsmatrizen (n_subjects x 200 x 200) -> Edge-Vektor
X = ConnectivityMeasure(kind="correlation").fit_transform(data.rois_cc200)
X = np.nan_to_num(X)  # leere ROIs einzelner Probanden
X = X[:, *np.triu_indices(X.shape[1], k=1)]  # oberes Dreieck -> n_subjects x n_edges

y_real = pheno["AGE_AT_SCAN"].astype(float).values
site = pheno["SITE_ID"]
confounds = pd.get_dummies(site, prefix="site", drop_first=True, dtype=float)
confounds = np.ascontiguousarray(confounds.values)

# --- Negativkontrolle: Alter innerhalb jedes Standorts permutieren
y_perm = y_real.copy()
for _, idx in pheno.groupby("SITE_ID").indices.items():
    idx = np.asarray(idx)
    y_perm[idx] = RNG.permutation(y_real[idx])

# --- Sanity Checks: Standortstruktur erhalten, individueller Zusammenhang zerstört
site_mean_before = pheno.groupby("SITE_ID")["AGE_AT_SCAN"].mean()
site_mean_after = pd.Series(y_perm, index=pheno.index).groupby(site).mean()
assert np.allclose(site_mean_before.sort_index(), site_mean_after.sort_index())
print(f"n = {len(y_real)}, {site.nunique()} Standorte")
print("Standortmittel (Alter) vor/nach Permutation identisch: OK "
      f"(max. Abweichung {np.abs(site_mean_before - site_mean_after).max():.2e})")
print(f"Individuelle Korrelation permutiertes vs. echtes Alter (within-site, erwartet ~0): "
      f"r = {np.corrcoef(y_real, y_perm)[0, 1]:.3f} (global, dominiert von between-site-Anteil)")


def run(name, covariates, corrected):
    cpm = CPMAnalysis(
        results_directory=f"results_abide_negative_control/{name}",
        task_type="regression",
        cv=RepeatedKFold(n_splits=5, n_repeats=5, random_state=42),
        edge_selection=UnivariateEdgeSelection(
            selection_statistic="pearson",
            selection_input="residualized" if corrected else "raw",
            edge_selection=[PThreshold(threshold=[0.01], correction=[None])]),
        model_input="residualized" if corrected else "raw",
        n_permutations=N_PERMUTATIONS, device=DEVICE)
    cpm.run(X, y_perm, covariates)
    return (pd.read_csv(f"results_abide_negative_control/{name}/cv_results_summary.csv",
                        header=[0, 1], index_col=[0, 1, 2]),
            pd.read_csv(f"results_abide_negative_control/{name}/p_values.csv",
                        index_col=[0, 1]))


naive_summary, naive_p = run("naive", None, corrected=False)
corrected_summary, corrected_p = run("corrected", confounds, corrected=True)

r_naive = naive_summary.loc[("connectome", "both", 0), ("pearson_score", "mean")]
p_naive = naive_p.loc[("both", "connectome"), "pearson_score"]
r_corrected = corrected_summary.loc[("connectome", "both", 0), ("pearson_score", "mean")]
p_corrected = corrected_p.loc[("both", "connectome"), "pearson_score"]

print("\nr (vorhergesagtes vs. permutiertes 'Alter', Mittel über CV-Folds; wahrer Effekt = 0):")
print(f"  Connectome, naiv           r = {r_naive:+.3f}  (p = {p_naive:.4f})")
print(f"  Connectome, Site-korrigiert r = {r_corrected:+.3f}  (p = {p_corrected:.4f})")
if p_naive < 0.05 <= p_corrected:
    print("  -> Fehlalarm bestätigt: naiv signifikant trotz Ground-Truth-Effekt = 0, "
          "nach Korrektur korrekt nicht signifikant.")
else:
    print("  -> Erwartetes Muster (naiv signifikant, korrigiert nicht) trat so nicht ein -- "
          "Zahlen oben genau prüfen.")
print("p-Werte und Abbildungen: results_abide_negative_control/*/p_values.csv und report.html")
