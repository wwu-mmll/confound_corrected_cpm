"""ABIDE I negative control: confound correction fixing a genuinely false result.

`abide_age_site_cpm.py` shows age prediction dropping after site correction
(r 0.564 -> 0.286), but real data can't prove that drop fixed a *wrong* result --
no one knows how much was real aging vs. scanner artifact. This script instead
builds a case with known ground truth: age is permuted *within* each site, which
destroys any individual connectome-age link while leaving each site's mean age --
and so the site-age relationship -- untouched. The true connectome-age effect is
now exactly zero by construction.

Expected: naive analysis still finds a "significant" effect (a false positive,
since site confounds both the connectome and the permuted age); the corrected
analysis finds none, because there is nothing real left to detect.
"""
import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
from nilearn.datasets import fetch_abide_pcp
from sklearn.model_selection import RepeatedKFold
from cccpm import CPMAnalysis, UnivariateEdgeSelection, PThreshold
import torch
import cccpm.reporting.data_insights as _di

# torch.backends.cuda.preferred_linalg_library("cusolver") only needed on turing generation gpus
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# cccpm's covariate pairplot is slow with ~20 site dummies
_di.pairplot_flexible = lambda df, path: None

N_PERMUTATIONS = 1000
RNG = np.random.default_rng(42)

data = fetch_abide_pcp(derivatives="rois_cc200", pipeline="cpac",
                       band_pass_filtering=True, quality_checked=True)
pheno = pd.DataFrame(data.phenotypic).reset_index(drop=True)

X = ConnectivityMeasure(kind="correlation").fit_transform(data.rois_cc200)
X = np.nan_to_num(X)  # a few subjects have empty ROIs
X = X[:, *np.triu_indices(X.shape[1], k=1)]

y_real = pheno["AGE_AT_SCAN"].astype(float).values
site = pheno["SITE_ID"]
confounds = pd.get_dummies(site, prefix="site", drop_first=True, dtype=float)
confounds = np.ascontiguousarray(confounds.values)

# permute age within site. age does no longer correlate with connectome after, but age still correlates with site
y_perm = y_real.copy()
for _, idx in pheno.groupby("SITE_ID").indices.items():
    idx = np.asarray(idx)
    y_perm[idx] = RNG.permutation(y_real[idx])

site_mean_before = pheno.groupby("SITE_ID")["AGE_AT_SCAN"].mean()
site_mean_after = pd.Series(y_perm, index=pheno.index).groupby(site).mean()
assert np.allclose(site_mean_before.sort_index(), site_mean_after.sort_index())
print(f"n = {len(y_real)}, {site.nunique()} sites")
print("Site means (age) identical before/after permutation: OK "
      f"(max deviation {np.abs(site_mean_before - site_mean_after).max():.2e})")
print(f"Individual correlation permuted vs. real age (within-site, expected ~0): "
      f"r = {np.corrcoef(y_real, y_perm)[0, 1]:.3f} (global, dominated by the between-site component)")


def run(name, covariates, corrected):
    """Run one cccpm analysis and load its summary and p-value results."""
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

print("\nr (predicted vs. permuted 'age', mean across CV folds; true effect = 0):")
print(f"  Connectome, naive           r = {r_naive:+.3f}  (p = {p_naive:.4f})")
print(f"  Connectome, site-corrected  r = {r_corrected:+.3f}  (p = {p_corrected:.4f})")
if p_naive < 0.05 <= p_corrected:
    print("  -> False positive confirmed: naive significant despite a ground-truth effect "
          "of 0, correctly non-significant after correction.")
else:
    print("  -> Expected pattern (naive significant, corrected not) did not occur -- "
          "check the numbers above.")
print("p-values and figures: results_abide_negative_control/*/p_values.csv and report.html")
