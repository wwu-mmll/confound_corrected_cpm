# Edge Statistics

The edge statistic itself: one vectorised OLS GLM covering Pearson and Spearman
and their covariate-controlled forms, plus ranks, residualisation and Bonferroni
correction.

This is what `UnivariateEdgeSelection(selection_statistic=..., selection_input=...)`
dispatches to — see [Confound control](../methods.md#2-confound-control) for which
combination means what.

::: cccpm.statistics
