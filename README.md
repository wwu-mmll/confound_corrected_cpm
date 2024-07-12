# Connectome-Based Predictive Modelling (The right way)
Python version of the Connectome-based Predictive Modelling framework

## Basic info
relevant paper: https://www.nature.com/articles/nprot.2016.178.pdf

## Notes
Implement a pipeline similar to sklearn or PHOTONAI that contains the first edge filtering (one sample ttest),
the edge statistic (pearson, spearman, partial) and the edge selection (mainly threshold).
If these steps are implemented in a pipeline, it might be easier to optimize each individual step. For this, we need
to implement a nested CV.
If the edge statistic is a multivariate model (e.g. Lasso), then edge statistic and edge selection is one step. Maybe
it might be could to merge these steps to one edge selection step and some methods also allow for a threshold parameter
to be optimized.