# Interpreting Your Results

After `cpm.run(...)` finishes, everything CCCPM produces is written to the
`results_directory` you specified. This page explains what each output means and
how to read it — both the interactive **HTML report** and the underlying CSV/array
files you can load for your own figures and statistics.

The same structure is produced for **regression** and **classification**; only the
performance metrics differ (see [Metrics](#metrics)).

---

## Two concepts that appear everywhere

Almost every result is broken down by a **model** and a **network**.

### Models — what each one is built from

For every cross-validation fold, CCCPM fits several models so you can see how much
the connectome actually adds on top of your covariates:

| Model | Predicts the target from… | Use it to answer… |
|-------|---------------------------|-------------------|
| `connectome` | connectome network strength only | How well does brain connectivity alone predict? |
| `covariates` | the covariates only (e.g. age, sex, motion) | What's the baseline from nuisance variables? |
| `full` | connectome **+** covariates | Best combined prediction |
| `residuals` | connectome, after removing covariate effects from the target | Connectome contribution with confounds removed |
| `increment` | the *added* value of the connectome over covariates alone (`full` − `covariates`) | Does the connectome explain anything **beyond** confounds? |

The `increment` model is usually the most important for a confound-aware claim: a
significant increment means the connectome carries predictive information that the
covariates do not.

### Networks — which edges are used

Edge selection splits predictive edges into two sub-networks, and CCCPM evaluates
each plus their combination:

- **positive** — edges *positively* correlated with the target.
- **negative** — edges *negatively* correlated with the target.
- **both** — positive and negative networks combined.

So a single number like "Pearson r" is always reported per *(model, network)* pair.

---

## The HTML report (`report.html`)

`report.html` is a self-contained, shareable summary — open it in any browser, print it
to PDF, or send it to a collaborator (everything, including the figures, is embedded, so
it works offline). It is a single scrolling page with a sticky sidebar, organised top-down
from the headline result to the supporting detail:

| Section | What it shows |
|---------|---------------|
| **Summary** | The headline cross-validated result as a one-sentence verdict, key-stat chips (samples, nodes, edges, covariates, permutations, edge p-threshold), and predicted-vs-observed scatter plots for the positive, negative, and both networks plus the covariates-only baseline. Each scatter is annotated with its cross-validated effect size (Pearson *r* / AUC) and permutation *p*. **Start here.** |
| **Model Comparison** | One faceted figure comparing every model (`connectome`, `covariates`, `full`, `residuals`, `increment`) across metrics and networks, plus the APA results table (mean [SD] with permutation *p*-values). Foregrounds the `increment` model — your confound-control evidence. |
| **Network Strengths** | How the summed positive/negative network strength relates to the target, and the distribution of strength across participants. |
| **Brain & Edges** | The predictive edges in the brain: a connectivity matrix, a hub (node-degree) plot, and — when an atlas with a `network` column is supplied — a network-summary matrix and a chord diagram. With node coordinates (`x, y, z`) it also renders a glass-brain view. |
| **Stable Edges** | The most reliably selected edges (region A — region B) sorted by significance, capped to the top rows per network. |
| **Data & Methods** | Appendix: target distribution, covariate scatter matrix, the full run configuration, and (with an inner CV) per-fold hyperparameters. |

Every section carries a short always-visible "what this shows" note so a reader unfamiliar
with the internals can follow it.

You can view a full example report in the
[Simulated Data example](examples/simulated_data.md).

---

## Metrics

Metrics are computed **out-of-sample** (on held-out test folds) and then averaged
across folds. Higher is better unless noted.

### Regression

| Metric | Meaning | Notes |
|--------|---------|-------|
| `pearson_score` | Correlation between predicted and true values | The conventional CPM headline number |
| `explained_variance_score` | Fraction of target variance explained | 1.0 is perfect; can be negative |
| `mean_absolute_error` | Average absolute prediction error | Lower is better; in target units |
| `mean_squared_error` | Average squared prediction error | Lower is better; penalises large errors |

### Classification

| Metric | Meaning | Notes |
|--------|---------|-------|
| `roc_auc` | Area under the ROC curve | Threshold-independent; robust headline metric |
| `balanced_accuracy` | Mean of sensitivity and specificity | **Prefer over accuracy for imbalanced classes** |
| `accuracy` | Fraction correctly classified | Misleading when classes are imbalanced |
| `f1_score` | Harmonic mean of precision and recall | Useful when the positive class is rare |

---

## Output files

Everything in the report is also available as plain files for your own analysis.

### Performance

| File | Contents |
|------|----------|
| `cv_results_summary.csv` | Mean ± std of every metric, indexed by `(model, network)`. The numbers behind the Main Results page. |
| `cv_results_full.csv` | Per-fold metric values (before averaging) — use these for your own error bars / tests. |
| `cv_predictions.csv` | Out-of-sample predictions per subject: columns `sample_index, model, network, y_pred, y_true, fold`. |
| `cv_network_strengths.csv` | Summed positive/negative network strength per subject, with the target: `y_true, network_strength, model, network, fold`. |

### Significance (only if `n_permutations > 0`)

| File | Contents |
|------|----------|
| `p_values.csv` | Permutation p-value for each `(network, model)` × metric. A small value (e.g. ≤ 0.05) means the observed performance is unlikely under the null of no brain–behaviour association. |
| `permutation/` | The full null distributions (`cv_results_*`) and permuted edge selections used to compute those p-values. |

**How the p-values are defined:** the entire pipeline is re-run on many shuffled
copies of the target to build a null distribution of performance; the p-value is the
fraction of permutations that match or beat the real result. With `n_permutations=100`
the smallest possible p-value is ~0.01 — use **1000+** permutations for publishable
significance.

### Selected edges & stability

| File | Contents |
|------|----------|
| `edges.npy` | The edges selected in each fold. |
| `stability_edges.npy` | How consistently each edge was selected across folds (its *stability*). Edges chosen in most folds are the reliable ones. |
| `stability_edges_significance.npy` | Significance of edge stability (when permutations are run). |
| `data_insights/` | Cached input summaries and figures (target distribution, covariate scatter, feature/target/covariate names). |

### Logs

- `cpm_log.txt` — a full, timestamped log of the run (configuration, per-step progress, final aggregated table). The first place to look if something behaved unexpectedly.
- `task_type.txt` — `regression` or `classification`, recorded for the report.
- `plots/` — every figure from the report exported as PNG/PDF/SVG for direct use in papers.

---

## A reading workflow

1. Open `report.html` → **Summary** for the headline result, then **Model Comparison**.
   Look at the `full` and `increment` models on the `both` network.
2. Check `p_values.csv`: is the `increment` model significant? That's your evidence the
   connectome adds information beyond confounds.
3. Inspect `stability_edges.npy` (or the **Stable Edges** section) to see *which* connections
   drive the prediction.
4. Use `cv_predictions.csv` / `cv_results_full.csv` to make your own figures or run
   additional statistics.
