# Getting Started

This guide will help you get started with running an analysis using the `CPMRegression` class. It provides a step-by-step description of how to set up, configure, and execute an analysis, along with explanations of the inputs and parameters.

---

## Step 1: Prepare Your Data

To run an analysis, you need the following inputs:

- **Connectome Data (`X`)**: A 2D array (numpy array or pandas DataFrame) of shape `(n_samples, n_features)` containing connectome edge values for each subject.
- **Target Variable (`y`)**: A 1D array or pandas Series of shape `(n_samples,)` containing the outcome variable (e.g., clinical scores, behavioral measures).
- **Covariates**: A 2D array or pandas DataFrame of shape `(n_samples, n_covariates)` containing variables to control for (e.g., age, sex).

Ensure that all inputs have consistent sample sizes (`n_samples`).

---

## Step 2: Configure the Analysis

### **Cross-Validation**
The `CPMRegression` class uses an outer cross-validation loop for performance evaluation and an optional inner cross-validation loop for hyperparameter optimization.

- **Outer CV (`cv`)**: Defines the cross-validation strategy (e.g., `KFold`).
- **Inner CV (`inner_cv`)**: Used for optimizing hyperparameters during edge selection. Can be left as `None` if not needed.

Example:

```python
from sklearn.model_selection import KFold

outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)
```

### Edge Selection
The toolbox implements univariate edge selection, allowing users to specify the method for evaluating and selecting edges based on statistical tests.

#### Edge Statistics
Choose from the following methods for computing edge statistics:

- **pearson**: Pearson correlation
- **pearson_partial**: Pearson partial correlation (controlling for covariates)
- **spearman**: Spearman rank correlation
- **spearman_partial**: Spearman partial correlation (controlling for covariates)

#### p-Thresholds
- Set a single value (e.g., 0.05) or provide multiple values (e.g., [0.01, 0.05, 0.1]).
- If multiple thresholds are specified, the toolbox will optimize for the best p-threshold during inner cross-validation.

#### FDR Correction
- Optional FDR correction for multiple comparisons can be applied using correction='fdr_by'.


Example:

```python
from cpm.edge_selection import UnivariateEdgeSelection, PThreshold

edge_statistic = 'pearson'
univariate_edge_selection = UnivariateEdgeSelection(
    edge_statistic=[edge_statistic],
    edge_selection=[PThreshold(threshold=[0.05], correction=['fdr_by'])]
)
```

## Step 3: Set Up the CPMRegression Object
Create an instance of the CPMRegression class with the required inputs:

```python
from cpm.models import CPMRegression

cpm = CPMRegression(
    results_directory="results/",
    cv=outer_cv,
    inner_cv=inner_cv,  # Optional
    edge_selection=univariate_edge_selection,
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
- **select_stable_edges**: Whether to select stable edges across folds (True or False).
- **stability_threshold**: Minimum proportion of folds in which an edge must be selected to be considered stable.
- **impute_missing_values**: Whether to impute missing values (True or False).
- **n_permutations**: Number of permutations for permutation testing.

## Step 4: Run the Analysis
Call the estimate method to perform the analysis:

```python
X = ...  # Load your connectome data (numpy array or pandas DataFrame)
y = ...  # Load your target variable (numpy array or pandas Series)
covariates = ...  # Load your covariates (numpy array or pandas DataFrame)

cpm.estimate(X=X, y=y, covariates=covariates)
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
By following these steps, you can quickly set up and execute a connectome-based predictive modeling analysis using the CPMRegression class. For further customization, refer to the API documentation.
