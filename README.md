[![GitHub Workflow Status](https://github.com/wwu-mmll/confound_corrected_cpm/actions/workflows/test.yml/badge.svg)](https://github.com/wwu-mmll/confound_corrected_cpm/actions/workflows/test.yml)
[![Coverage Status](https://coveralls.io/repos/github/wwu-mmll/confound_corrected_cpm/badge.svg)](https://coveralls.io/github/wwu-mmll/confound_corrected_cpm)
[![Github Contributors](https://img.shields.io/github/contributors-anon/wwu-mmll/cpm_python?color=blue)](https://github.com/wwu-mmll/cpm_python/graphs/contributors)
[![Github Commits](https://img.shields.io/github/commit-activity/y/wwu-mmll/cpm_python)](https://github.com/wwu-mmll/cpm_python/commits/main)

# Confound-Corrected Connectome-Based Predictive Modelling in Python
**Confound-Corrected Connectome-Based Predictive Modelling** is a Python package for performing connectome-based predictive modeling (CPM). This toolbox is designed for researchers in neuroscience and psychiatry, providing robust methods for building predictive models based on structural or functional connectome data. It emphasizes replicability, interpretability, and flexibility, making it a valuable tool for analyzing brain connectivity and its relationship to behavior or clinical outcomes.

---

## What is Connectome-Based Predictive Modeling?

Connectome-based predictive modeling (CPM) is a machine learning framework that leverages the brain's connectivity patterns to predict individual differences in behavior, cognition, or clinical status. By identifying key edges in the connectome, CPM creates models that link connectivity metrics with target variables (e.g., clinical scores). This approach is particularly suited for studying complex relationships in neuroimaging data and developing interpretable predictive models.

---

## Key Features

- **Univariate Edge Selection**: Supports methods like `pearson`, `spearman`, and their partial correlation counterparts, with options for p-threshold optimization and FDR correction.
- **Cross-Validation**: Implements nested cross-validation for robust model evaluation.
- **Edge Stability**: Selects stable edges across folds to improve model reliability.
- **Confound Adjustment**: Controls for covariates during edge selection and modeling.
- **Permutation Testing**: Assesses the statistical significance of models using robust permutation-based methods.

---

## Documentation

For detailed instructions on installation, usage, and advanced configurations, visit the [documentation website](https://wwu-mmll.github.io/confound_corrected_cpm/).

---

## Installation

Install the package from GitHub:

```bash
git clone https://github.com/mmll/confound_corrected_cpm.git
cd cpm_python
pip install .
```

## Quick Example
Here's a quick overview of how to run a CPM analysis:

```python
from src.cccpm.cpm_analysis import CPMRegression
from src.cccpm.edge_selection import UnivariateEdgeSelection, PThreshold
from sklearn.model_selection import KFold

# Configure edge selection
univariate_edge_selection = UnivariateEdgeSelection(
    edge_statistic=["pearson"],
    edge_selection=[PThreshold(threshold=[0.05], correction=["fdr_by"])]
)

# Create the CPMRegression object
cpm = CPMRegression(
    results_directory="results/",
    cv=KFold(n_splits=10, shuffle=True, random_state=42),
    edge_selection=univariate_edge_selection,
    n_permutations=100
)

# Run the analysis
X = ...  # Connectome data
y = ...  # Target variable
covariates = ...  # Covariates
cpm.run(X, y, covariates)
```

## Contributing
Contributions are welcome! If you have ideas, feedback, or feature requests, feel free to open an issue or submit a pull request on the GitHub repository.

