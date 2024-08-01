[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/wwu-mmll/cpm_python/cpm/python-test)](https://github.com/wwu-mmll/cpm_python/actions)
[![Coverage Status](https://coveralls.io/repos/github/wwu-mmll/cpm_python/badge.svg?branch=main)](https://coveralls.io/github/wwu-mmll/cpm_python?branch=main)
[![Github Contributors](https://img.shields.io/github/contributors-anon/wwu-mmll/cpm_python?color=blue)](https://github.com/wwu-mmll/cpm_python/graphs/contributors)
[![Github Commits](https://img.shields.io/github/commit-activity/y/wwu-mmll/cpm_python)](https://github.com/wwu-mmll/cpm_python/commits/main)

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

### Hyperparameters
- edge statistic and selection
    - univariate
        - correlation (pearson, spearman, semi-partial)
        - mutual information
        - simple p-value, FDR, FWE, FPR, k-best, percentile
            - absolute p-threshold (p)
            - corrected p-threshold: FDR, FWE, FPR
            - relative (k-best, percentile)
    - multivariate
        - lasso (L1)
        - tree-based
        - recursive feature elimination
    - use stability over inner CV folds (only those edges that are selected for within all inner CV folds)
        - could apply for both multivariate and univariate

- predictive model
    - use covariates: yes/no
    -
