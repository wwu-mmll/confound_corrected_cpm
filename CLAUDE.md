# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Confound-Corrected Connectome-based Predictive Modeling (CCCPM) — a Python package for CPM analysis supporting regression and binary classification. Uses PyTorch for vectorized computation with CUDA/CPU support.

## Build & Development Commands

```bash
# Install dependencies
poetry install

# Run all tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=cccpm --cov-report=term

# Run a single test file
poetry run pytest tests/test_scoring.py

# Run a single test
poetry run pytest tests/test_scoring.py::test_function_name -v

# Build package
poetry build

# Build documentation
mkdocs build
```

## Architecture

### Pipeline Flow

`CPMAnalysis.run()` orchestrates the full pipeline:
1. **Data validation** — `check_data()`, `impute_missing_values()` in `utils.py`
2. **Task type detection** — auto-detects regression vs classification from target variable
3. **Outer CV loop** — splits data, runs edge selection → model fitting → scoring per fold
4. **Inner CV** (optional) — hyperparameter tuning via `run_inner_folds()` in `inner_fold.py`
5. **Permutation testing** (optional) — shuffled targets for statistical significance
6. **Results aggregation** — `ResultsManager` / `PermutationManager` in `results_manager.py`
7. **HTML report** — `reporting/html_report.py`

### Key Modules

| Module | Role |
|--------|------|
| `cpm_analysis.py` | Main `CPMAnalysis` class — entry point and orchestrator |
| `pytorch_model.py` | `LinearCPMModel` — PyTorch linear/logistic regression with Cholesky solver |
| `edge_selection.py` | `UnivariateEdgeSelection` — correlation-based feature selection (Pearson/Spearman/partial) with p-value thresholding |
| `scoring.py` | `FastCPMMetrics` / `FastCPMClassificationMetrics` — GPU-accelerated metrics |
| `inner_fold.py` | Inner CV for hyperparameter optimization |
| `results_manager.py` | `ResultsManager` / `PermutationManager` — aggregation and p-value computation |
| `constants.py` | Enums: `TaskType`, `Networks`, `Models`, `Metrics` |
| `utils.py` | Data validation, train/test splitting, edge stability, matrix/vector conversion |

### Internal Tensor Shapes

- Input X: `[N_samples, N_features]`
- Target y: `[N_samples, 1]` (or `[N_samples, N_runs]` for permutations)
- Edges: `[N_features, 2, N_runs]` (dim 1: positive/negative networks)
- Predictions: `[N_samples, N_models, N_networks, N_runs]`
- Metrics: `[N_metrics, N_models, N_networks, N_runs]`

### Model Variants

Each fold fits four model types (defined in `Models` enum): **connectome**, **covariates**, **full**, **residuals**. Each is evaluated across network types (positive, negative, both).

### Package Structure

Source code lives in `src/cccpm/` (Poetry src layout). Tests in `tests/` with fixtures in `conftest.py` providing simulated data. The package is importable as `cccpm`.

## CI

GitHub Actions runs `pytest --cov` on push/PR to `main` and `develop` branches (Python 3.11).
