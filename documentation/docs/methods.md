# How CCCPM Works

This page explains the method behind CCCPM so you can configure an analysis with
intent and interpret it correctly. For a runnable walkthrough, see the
[regression](examples/regression.md) and [classification](examples/classification.md)
examples; for the outputs, see [Interpreting Results](interpreting_results.md).

## The CPM idea in one paragraph

Connectome-based predictive modeling (CPM) builds an interpretable model that links
brain connectivity to an outcome. For each connection (*edge*) it measures how
strongly that edge relates to the target, keeps the most related edges, summarizes
them into a small number of *network strength* scores, and fits a simple model from
those scores to the target. Because the final model is low-dimensional and the
selected edges are inspectable, CPM is both predictive and interpretable.

## The pipeline

CCCPM runs the following inside a cross-validation loop so that performance is always
estimated on held-out data.

### 1. Edge selection

Each edge is correlated with the target, and edges below a p-value threshold are kept
and split into a **positive** network (edges that increase with the target) and a
**negative** network (edges that decrease with it). Available statistics:

| Statistic | For | Controls for covariates? |
|-----------|-----|--------------------------|
| `pearson` / `spearman` | continuous target | no |
| `pearson_partial` / `spearman_partial` | continuous target | yes (during selection) |
| `point_biserial` | binary target | no |
| `point_biserial_partial` | binary target | yes (during selection) |

Use the `*_partial` variants to remove covariate effects **while selecting edges**, so
that selection is not driven by confounds.

### 2. Confound control

CCCPM offers two complementary ways to control for nuisance variables:

- **Partial correlation during edge selection** (the `*_partial` statistics above).
- **Residualization** (`calculate_residuals=True`): regress the covariates out of the
  connectome before modeling.

In addition, the covariates are always available to the model variants below, so you
can quantify what the connectome adds *beyond* them.

### 3. Model variants

For each fold, CCCPM fits several models so you can separate the connectome's
contribution from the covariates':

- `connectome` — network strengths only.
- `covariates` — covariates only (a baseline).
- `full` — connectome + covariates.
- `residuals` — connectome predicting the covariate-residualized target.
- `increment` — the added value of the connectome over covariates (`full` − `covariates`).

Each is evaluated on the positive, negative, and combined networks.

### 4. Nested cross-validation (optional)

An **outer** CV loop estimates unbiased performance. An optional **inner** CV loop
tunes hyperparameters (most importantly the p-threshold) without leaking test data —
pass an `inner_cv` and multiple thresholds to enable it.

### 5. Edge stability

When edges are selected repeatedly across folds, CCCPM records how often each edge is
chosen — its **stability**. Edges selected in most folds are the reliable ones, and
`select_stable_edges=True` restricts the model to edges above `stability_threshold`.

### 6. Permutation testing

With `n_permutations > 0`, the whole pipeline is re-run on many shuffled copies of the
target to build a null distribution of performance. The resulting p-value is the
fraction of permutations that match or beat the real result — your evidence that the
brain–behaviour association is not due to chance. Use **1000+** permutations for
publishable significance.

## Putting it together

A typical confound-aware analysis:

1. Choose a partial edge statistic (`pearson_partial`, or `point_biserial_partial` for
   classification) to control covariates during selection.
2. Use an inner CV to tune the p-threshold.
3. Enable permutation testing.
4. Read the `increment` model's significance to claim the connectome adds information
   **beyond** your covariates.

See [Interpreting Results](interpreting_results.md) for how to read every output.
