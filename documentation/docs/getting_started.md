# Getting Started

This guide will help you get started with running an analysis using the `CPMAnalysis` class. It provides a step-by-step description of how to set up, configure, and execute an analysis, along with explanations of the inputs and parameters. `CPMAnalysis` supports both **regression** and **binary classification**; the task type is auto-detected from your target variable or can be set explicitly with `task_type`.

---

## Step 1: Prepare Your Data

To run an analysis, you need the following inputs:

- **Connectome Data (`X`)**: A 2D array (numpy array or pandas DataFrame) of shape `(n_samples, n_features)` containing connectome edge values for each subject.
- **Target Variable (`y`)**: A 1D array or pandas Series of shape `(n_samples,)` containing the outcome variable (e.g., clinical scores, behavioral measures).
- **Covariates** *(optional)*: A 2D array or pandas DataFrame of shape `(n_samples, n_covariates)` containing variables to control for (e.g., age, sex). Omit it for vanilla CPM — only the `connectome` model is then defined, and the covariate-dependent rows are NaN-filled rather than silently invented.

Ensure that all inputs have consistent sample sizes (`n_samples`).

---

## Step 2: Configure the Analysis

### **Cross-Validation**
The `CPMAnalysis` class uses an outer cross-validation loop for performance evaluation and an optional inner cross-validation loop for hyperparameter optimization.

- **Outer CV (`cv`)**: Defines the cross-validation strategy (e.g., `KFold`).
- **Inner CV (`inner_cv`)**: Used for optimizing hyperparameters during edge selection. Can be left as `None` if not needed.

Example:

```python
from sklearn.model_selection import KFold

outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)
```

### Edge Selection
The toolbox implements univariate edge selection, allowing users to specify the method for evaluating and selecting edges based on statistical tests.

#### Edge statistic and confound control

Edge selection has two independent settings:

- **`selection_statistic`** — `'pearson'` or `'spearman'` (default). A binary
  target needs no separate statistic: point-biserial correlation *is* Pearson
  against a 0/1 outcome.
- **`selection_input`** — `'raw'` (default) or `'residualized'`. With
  `'residualized'`, each edge is tested by the regression `y ~ 1 + Z + edge`:
  the reported effect size is the semipartial correlation and the p-value is the
  edge coefficient's, with `df = N − 2 − C`. This is how you stop covariates
  from driving *which* edges get selected.

`selection_input` is a design decision, not a hyperparameter — it never enters
the inner CV's parameter grid.

#### p-Thresholds
- Set a single value (e.g., 0.05) or provide multiple values (e.g., [0.01, 0.05, 0.1]).
- If multiple thresholds are specified, the toolbox will optimize for the best p-threshold during inner cross-validation.

#### FDR Correction
- Optional FDR correction for multiple comparisons can be applied using correction='fdr_by'.


Example:

```python
from cccpm import UnivariateEdgeSelection, PThreshold

univariate_edge_selection = UnivariateEdgeSelection(
    selection_statistic='pearson',
    selection_input='residualized',     # selection controls for the covariates
    edge_selection=[PThreshold(threshold=[0.05], correction=['fdr_by'])]
)
```

#### Deconfounding the connectome the model sees

`selection_input` governs *selection*. The second, independent choice is
`model_input` on `CPMAnalysis` — whether the covariate variance is regressed out
of the connectome the models consume. It is fitted on the training fold, applied
to the test fold, and applied **after** edge selection, so the two never
interfere.

```python
CPMAnalysis(..., model_input='residualized')
```

Together they form a 2×2, and each cell is a separate run. There is no
`residuals` model: deconfounding is a property of the run, because the
non-linear backends are not invariant to it and the name would mean something
different for each of them. See [How CCCPM Works](methods.md#2-confound-control).

### Edge & Network Significance

When permutation testing is enabled (`n_permutations > 0`), CCCPM tests whether an edge is
selected across folds **more consistently than chance** by building a null distribution of
edge *stability* (the fraction of folds an edge is selected) from the shuffled-target runs.

Because there are tens of thousands of edges — and permutation p-values are floored at
`1/(n_permutations + 1)` — per-edge FDR is effectively powerless and a per-edge
max-statistic is crippled by the discreteness of stability. CCCPM therefore controls the
family-wise error rate at the **subnetwork** level, chosen with
`stability_significance_method`.

!!! warning "Two different significances"
    `stability_significance_method` is about **stability** — whether an edge is
    selected across folds more often than chance. It has nothing to do with the
    p-value that decides whether an edge is selected in the first place, which
    lives on `PThreshold(threshold=...)`. The parameter was called
    `edge_significance_method` before 0.7.0, which ran the two together;
    likewise `nbs_threshold` is now `nbs_stability_threshold`, because it is a
    fraction of folds, not a p-value. Both old spellings raise and name their
    replacement.


- **`"nbs"` (default)** — the [Network-Based Statistic](https://doi.org/10.1016/j.neuroimage.2010.06.041)
  (Zalesky et al., 2010). Edges whose stability meets `nbs_stability_threshold` form a graph; its
  connected components are tested against a permutation null of the **largest component**.
  A significant result means *this connected subnetwork* is selected more consistently than
  chance — it is **not** a claim about any single edge in isolation.
- **`"tfce"`** — network Threshold-Free Cluster Enhancement: per-edge FWER-corrected
  p-values with no arbitrary cluster-forming threshold.

```python
cpm = CPMAnalysis(
    results_directory="results/",
    cv=outer_cv,
    edge_selection=univariate_edge_selection,
    n_permutations=1000,
    stability_significance_method="nbs",   # or "tfce"
    nbs_stability_threshold=0.5,      # fraction of folds, NOT a p-value
    nbs_component_stat="extent",      # "extent" (edge count) or "intensity"
)
```

The results are written to `stability_edges_significance.npy` (per-edge p-values; with NBS
every edge in a significant subnetwork shares that subnetwork's p-value) and
`stability_edges_significance_meta.json` (method, parameters, the permutation null
distribution, and the observed components). The HTML report visualises these in the
**Stable Edges** section — the significance method, the largest significant subnetwork, the
null-distribution plots, every significant edge, and a downloadable CSV of all selected
edges. See [How CCCPM Works](methods.md#7-edge-stability-significance) for the statistical
details.

## Step 3: Set Up the CPMAnalysis Object
Create an instance of the `CPMAnalysis` class with the required inputs:

```python
from cccpm import CPMAnalysis

cpm = CPMAnalysis(
    results_directory="results/",
    cv=outer_cv,
    inner_cv=inner_cv,  # Optional
    edge_selection=univariate_edge_selection,
    model_input="residualized",   # deconfound the connectome the models consume
    select_stable_edges=True,
    stability_threshold=0.8,
    impute_missing_values=True,
    n_permutations=100
)
```
### Key Parameters
- **results_directory**: Directory where results will be saved.
- **cv**: Outer cross-validation strategy.
- **inner_cv**: Inner cross-validation strategy for hyperparameter optimization (optional).
- **edge_selection**: Configuration for univariate edge selection.
- **model_input**: `'raw'` (default) or `'residualized'` — whether the connectome handed to the models has the covariate variance regressed out. Independent of `selection_input`. See [Confound control](methods.md#2-confound-control).
- **select_stable_edges**: Whether to select stable edges across folds (True or False).
- **stability_threshold**: Minimum proportion of folds in which an edge must be selected to be considered stable.
- **impute_missing_values**: Whether to impute missing values (True or False).
- **n_permutations**: Number of permutations for permutation testing.
- **stability_significance_method**: How edge-*stability* significance is established from the permutations — `"nbs"` (default, subnetwork-level) or `"tfce"` (per-edge). Not related to the edge-selection p-value on `PThreshold`. See [Edge & Network Significance](#edge-network-significance).
- **nbs_stability_threshold**: Stability threshold (`>=`) for NBS component forming — a fraction of folds, default `0.5`.
- **nbs_component_stat**: NBS component statistic — `"extent"` (edge count, default) or `"intensity"` (summed supra-threshold stability).
- **atlas** *(optional)*: A built-in atlas name (e.g. `"Schaefer100-17"`) or a path to a custom CSV of region names and MNI coordinates, used for the brain figures in the report. See [Brain Atlases](atlases.md).

## Step 4: Run the Analysis
Call the `run` method to perform the analysis:

```python
X = ...  # Load your connectome data (numpy array or pandas DataFrame)
y = ...  # Load your target variable (numpy array or pandas Series)
covariates = ...  # Load your covariates (numpy array or pandas DataFrame)

cpm.run(X=X, y=y, covariates=covariates)

# ...or, for vanilla CPM with no covariates at all:
cpm.run(X=X, y=y)
```

This will:

1. Perform edge selection based on the specified method and thresholds.
2. Train and evaluate models for each cross-validation fold.
3. Save results, including predictions, metrics, and permutation-based significance tests, to the results_directory.


## Step 5: Review Results
After the analysis, you can find the results in the results_directory, including:

- Cross-validation metrics (e.g., mean absolute error, R²).
- Model predictions for each fold.
- Edge stability and significance.

You can load and inspect these results for further analysis.

---
By following these steps, you can quickly set up and execute a connectome-based predictive modeling analysis using the `CPMAnalysis` class. For further customization, refer to the API documentation.
