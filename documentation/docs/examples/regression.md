# Regression Example

This walkthrough runs a complete connectome-based predictive modeling (CPM)
analysis for a **continuous** target (regression) on simulated data. The script
below is the runnable file `examples/regression_quickstart.py` — copy it and swap in
your own data.

## The full script

```python
--8<-- "examples/regression_quickstart.py"
```

## Step by step

**1. Data.** CCCPM expects three aligned arrays:

- `X` — the connectome, shape `(n_samples, n_features)`, where `n_features` are the
  upper-triangular edges of a symmetric node-by-node matrix
  (`n_features = n_nodes * (n_nodes - 1) / 2`). You can also pass full matrices of
  shape `(n_samples, n_nodes, n_nodes)` and CCCPM will vectorize them.
- `y` — the continuous outcome, shape `(n_samples,)`.
- `covariates` — nuisance variables to control for (e.g. age, sex, motion).

**2. Edge selection.** `UnivariateEdgeSelection` correlates each edge with the target
and keeps edges below a p-value threshold. For regression, use `pearson` or
`spearman`; switch to the `*_partial` variants (e.g. `pearson_partial`) to control for
covariates *during* edge selection.

**3. Run.** `CPMAnalysis` performs the cross-validated fit, and — because
`n_permutations > 0` — permutation testing for significance. For a real analysis,
prefer `1000+` permutations.

**4. Results.** Everything is written under `results_directory`, including
`report.html`. See [Interpreting Results](../interpreting_results.md) for a full guide
to the metrics, model variants, and output files.

## See also

- [Classification example](classification.md) — the same workflow for a binary target.
- [Interpreting Results](../interpreting_results.md) — what every output means.
