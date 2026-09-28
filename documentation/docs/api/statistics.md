# Edge Statistics

The edge statistic itself: one vectorised OLS GLM covering Pearson and Spearman
and their covariate-controlled forms, plus ranks, residualisation and the exact
t-distribution p-values of the edge coefficient.

p-values are exact. The pipeline does not compute one per edge — that is a CPU
call per edge and permutation — but thresholds the t statistic against the
exact critical value of each p-threshold (`critical_t`), which selects the same
edges.

This is what `UnivariateEdgeSelection(selection_statistic=..., selection_input=...)`
dispatches to — see [Confound control](../methods.md#2-confound-control) for which
combination means what.

::: cccpm.statistics
