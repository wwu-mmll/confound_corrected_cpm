# Classification Example

This walkthrough runs a complete CPM analysis for a **binary** target
(classification) on simulated data. It mirrors the
[regression example](regression.md) — the workflow is identical; only the target,
edge statistic, cross-validation splitter, and reported metrics differ. The script
below is the runnable file `examples/classification_quickstart.py`.

## The full script

```python
--8<-- "examples/classification_quickstart.py"
```

## What differs from regression

| Aspect | Regression | Classification |
|--------|------------|----------------|
| Target `y` | continuous | binary (0/1) |
| `task_type` | `"regression"` | `"classification"` (or `None` to auto-detect) |
| `selection_statistic` | `pearson` / `spearman` | `pearson` — point-biserial correlation *is* Pearson against a 0/1 target, so there is nothing extra to select |
| Cross-validation | `KFold` | `StratifiedKFold` (keeps class balance across folds) |
| Metrics | explained variance, Pearson r, MSE, MAE | accuracy, balanced accuracy, F1, ROC AUC |
| `increment` reported for | explained variance, MSE, MAE (not Pearson r) | accuracy, balanced accuracy, ROC AUC (not F1) |

!!! tip
    For imbalanced classes, prefer **balanced accuracy** and **ROC AUC** over plain
    accuracy. CCCPM reports all four so you can compare.

## Step by step

The steps are the same as in the [regression example](regression.md): prepare your
`X` / `y` / `covariates`, configure `UnivariateEdgeSelection`, construct `CPMAnalysis`,
and call `run`. The target here is binary, so we use a `StratifiedKFold` splitter and
set `task_type="classification"`. Confound control works exactly as in regression —
`selection_input` for edge selection, `model_input` for the connectome the models
consume.

## Results

Everything is written under `results_directory`, including `report.html` with
classification metrics. See [Interpreting Results](../interpreting_results.md) for a
full guide.

## See also

- [Regression example](regression.md) — the same workflow for a continuous target.
- [Interpreting Results](../interpreting_results.md) — what every output means.
