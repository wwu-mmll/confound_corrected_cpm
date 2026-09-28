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

| Choice | Values | What it decides |
|--------|--------|-----------------|
| `selection_statistic` | `"pearson"`, `"spearman"` (default) | which correlation measures edge–target association |
| `selection_input` | `"raw"` (default), `"residualized"` | whether selection controls for the covariates |

A **binary target needs no special statistic**: point-biserial correlation *is*
Pearson against a 0/1 outcome, which is exactly what `"pearson"` computes.

`selection_input="residualized"` fits one regression per edge, `y ~ 1 + Z + edge`.
It reports the **semipartial** correlation as the effect size and the edge
coefficient's p-value, with `df = N − 2 − C` for `C` covariates. This is a
deliberate design decision reported in your methods section, not something the
inner CV tunes per fold — it never enters the parameter grid.

```python
UnivariateEdgeSelection(
    selection_statistic="pearson",
    selection_input="residualized",      # selection controls for covariates
    edge_selection=[PThreshold(threshold=[0.05], correction=[None])],
)
```

!!! note "Why a coefficient test rather than correlating residualized edges"
    Correlating residualized edges against the target does not hold its nominal
    alpha: the more confounded the data, the more the effective threshold
    shrinks, so `threshold=0.05` stops meaning 0.05. The per-edge coefficient
    test keeps the threshold nominal whatever the confounding.

#### Dropping structural zeros (sparse connectomes)

For sparse structural connectomes (e.g. DTI streamline counts), many edges are zero
for most subjects. The optional **presence filter** removes them before selection:

```python
UnivariateEdgeSelection(selection_statistic="pearson", presence_filter=0.5)
```

`presence_filter` keeps an edge only if it is nonzero in at least the given fraction of
subjects (`True` = `0.5`, the majority; a float sets the fraction). It is computed per
fold on the training subjects from the connectome alone, so it adds no target leakage,
and it is *additive* to the built-in near-zero-variance gate. It is **not** a test of
the mean against zero — an edge present in a small but consistent minority is *dropped*,
by design. Leave it off (`False`, the default) for functional data, whose edges have a
real signed distribution around a mean of ~0. The filter always sees the raw
connectome: `model_input="residualized"` is applied *after* selection, so the two
choices stay independent.

#### Keeping only connected edges

```python
UnivariateEdgeSelection(selection_statistic="pearson", connected_components=True)
```

`connected_components` keeps only edges that belong to a connected component (per
positive/negative network) with at least a minimum number of edges, dropping isolated
single edges. `True` uses a minimum of 2 edges (drop lone edges); an int sets the
minimum explicitly. Applied per fold (and per permutation) after thresholding, this
favours coherent subnetworks over scattered one-off edges and can improve edge
stability.

### 2. Confound control

Confound control is **two independent run-level choices**, not a menu of model
variants:

| Choice | Where | Question it answers |
|--------|-------|---------------------|
| `selection_input` | `UnivariateEdgeSelection` | does edge *selection* control for the covariates? |
| `model_input` | `CPMAnalysis` | is the covariate variance regressed out of the *connectome the models consume*? |

`model_input="residualized"` fits the residualization on the training fold and
applies it to the test fold, **after** edge selection — so the two axes never
interfere. Together they form a 2×2, and each cell is its own run:

```python
CPMAnalysis(
    results_directory="results/controlled",
    edge_selection=UnivariateEdgeSelection(
        selection_statistic="pearson",
        selection_input="residualized",          # axis 1
        edge_selection=[PThreshold(threshold=[0.05], correction=[None])]),
    model_input="residualized",                  # axis 2
)
```

!!! info "Why deconfounding is a run-level choice, not a model called `residuals`"
    OLS is invariant to it once the covariates are in the design — residualizing
    moves variance from the strength column into the `Z` columns, which are
    already there — so `full` and `increment` do not move. The non-linear
    backends are not invariant: on identical edges, `full` shifts by 391% of
    sd(y) for `DecisionTreeCPM`, 65% for `RandomForestCPM` and 19% for `GAMCPM`.
    A model *name* would therefore mean something different for every backend,
    and you could not tell which connectome produced `full`. Making it a
    property of the run removes the ambiguity.

Covariates are always available to the model variants below, so you can quantify
what the connectome adds *beyond* them.

#### Running without covariates

`covariates` is optional. Passing none is vanilla CPM: only `connectome` is
defined, and `covariates` / `full` / `increment` are NaN-filled — the results
keep their full shape, and `available_models.json` tells the report which rows
carry a real number. Options that presuppose covariates
(`selection_input="residualized"`, `model_input="residualized"`) raise up front,
naming the offending parameter, rather than silently degrading into a different
analysis.

### 3. Model variants

For each fold, CCCPM fits several models so you can separate the connectome's
contribution from the covariates':

- `connectome` — network strengths only.
- `covariates` — covariates only (a baseline).
- `full` — connectome + covariates.
- `increment` — the added value of the connectome over covariates (`full` − `covariates`).

Each is evaluated on the positive, negative, and combined networks. Whether the
connectome behind `connectome` and `full` has been deconfounded is a property of
the run (`model_input`), not a separate model name.

`increment` is reported as **NaN for Pearson *r* and F1**, and this is
deliberate: a difference of two correlations is not itself a correlation (that
needs Fisher *z* / Steiger's test), and a difference of two F1 scores is not an
F1. It is available for explained variance, MSE, MAE, accuracy, balanced
accuracy and ROC AUC.

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

### 7. Edge-stability significance

The same permutations also give a null distribution of edge **stability**, so CCCPM can
test whether an edge is selected across folds *more consistently than chance*. Because
there are tens of thousands of edges and permutation p-values are floored at
`1/(n_perm+1)`, per-edge FDR is hopeless and a per-edge max-statistic is crippled by the
discreteness of stability. CCCPM instead controls the family-wise error rate at the
**subnetwork** level, selected via `stability_significance_method`:

- **`"nbs"` (default)** — the Network-Based Statistic (Zalesky et al., 2010). Edges above
  `nbs_stability_threshold` form a graph; its connected components are tested against a permutation
  null of the largest component (size = `nbs_component_stat="extent"`, or summed
  supra-threshold stability = `"intensity"`). A significant result means *this connected
  subnetwork* is selected more consistently than chance — not a claim about any single
  edge.
- **`"tfce"`** — network Threshold-Free Cluster Enhancement, giving per-edge FWER-corrected
  p-values with no arbitrary cluster-forming threshold.

Both write `stability_edges_significance.npy` (per-edge p-values; NBS edges carry their
subnetwork's p-value) and a `stability_edges_significance_meta.json` with the null
distribution and component diagnostics that the HTML report visualises.

## Putting it together

A typical confound-aware analysis:

1. Set `selection_input="residualized"` so covariates cannot drive which edges
   are selected.
2. Set `model_input="residualized"` so they cannot drive the model either. (For
   the linear backend this leaves `full` and `increment` unchanged — it is the
   `connectome` model it protects.)
3. Use an inner CV to tune the p-threshold.
4. Enable permutation testing.
5. Read the `increment` model's significance to claim the connectome adds
   information **beyond** your covariates — on a metric where `increment` is
   defined (explained variance, not Pearson *r*).

See [Interpreting Results](interpreting_results.md) for how to read every output.
