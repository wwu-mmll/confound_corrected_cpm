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
1. **Data validation** — `check_data()` in `validation.py`, imputation/residualisation in `preprocessing.py`
2. **Task type detection** — auto-detects regression vs classification from target variable
3. **Outer CV loop** — splits data, runs edge selection → model fitting → scoring per fold
4. **Inner CV** (optional) — hyperparameter tuning via `run_inner_folds()` in `inner_fold.py`
5. **Permutation testing** (optional) — shuffled targets for statistical significance
6. **Results aggregation** — `ResultsManager` in `results_manager.py`; permutation p-values, NBS/TFCE in `inference.py`
7. **HTML report** — `reporting/html_report.py`

### Key Modules

| Module | Role |
|--------|------|
| `cpm_analysis.py` | Main `CPMAnalysis` class — entry point and orchestrator |
| `models/linear_model.py` | `LinearCPM` — PyTorch linear/logistic regression with Cholesky solver |
| `models/nonlinear_models.py` | `DecisionTreeCPM` / `RandomForestCPM` / `GAMCPM` — alternative CPM model backends |
| `statistics.py` | The edge statistic itself — one vectorised OLS GLM covering Pearson/Spearman/point-biserial and their partial variants, plus ranks, residualisation and Bonferroni |
| `edge_selection.py` | `UnivariateEdgeSelection` / `PThreshold` / `EdgeStatistic` — the `selection_statistic` x `selection_input` confound choice, p-value thresholding, presence and connected-component filters, parameter grid |
| `scoring.py` | `FastCPMMetrics` / `FastCPMClassificationMetrics` — GPU-accelerated metrics |
| `inner_fold.py` | Inner CV for hyperparameter optimization |
| `results_manager.py` | `ResultsManager` — preallocated accumulation of per-fold results |
| `inference.py` | `PermutationManager` — permutation p-values, NBS and TFCE edge-level correction |
| `constants.py` | Enums: `TaskType`, `Networks`, `Models`, `Metrics` |
| `validation.py` | Input validation (`check_data`), task-type detection, variable names |
| `connectome.py` | Matrix <-> upper-triangular-vector conversion, the single place the edge indexing convention lives |
| `preprocessing.py` | Per-fold train/test split, imputation, confound residualisation, edge-stability thresholding |
| `memory.py` | Permutation chunk planning (`plan_permutation_chunk`) |
| `reporting/data_insights.py` | Input-data summary figures — kept in `reporting/` so the numeric core stays plotting-free |

### Internal Tensor Shapes

- Input X: `[N_samples, N_features]`
- Target y: `[N_samples, 1]` (or `[N_samples, N_runs]` for permutations)
- Edges: `[N_features, 2, N_runs]` (dim 1: positive/negative networks)
- Predictions: `[N_samples, N_models, N_networks, N_runs]`
- Metrics: `[N_metrics, N_models, N_networks, N_runs]`

### Model Variants

Each fold fits four model types (defined in `Models` enum): **connectome**, **covariates**, **full**, **connectome_residualized**, plus **increment** (full − covariates) computed at aggregation. `connectome_residualized` is the connectome model with the covariate variance removed from the features; it is computed at the network-strength level, which is provably identical to residualising the edges (verified 4.8e-07) and far cheaper. Each is evaluated across network types (positive, negative, both).

Passing `covariates=None` to `CPMAnalysis.run` is vanilla CPM: only **connectome** is
defined, the other variants are NaN-filled (the results tensor keeps its full shape),
and `available_models.json` in the results directory tells the report which model rows
carry a real number. Options that presuppose covariates (`*_partial` edge statistics,
`selection_input='residualized'`) raise up front.

### Confound control

Two independent choices, not four levers:

- **`selection_input`** (`'raw'` | `'residualized'`, on `UnivariateEdgeSelection`) — does
  edge selection control for the covariates? `'residualized'` is one regression per edge,
  `y ~ 1 + Z + edge`: semipartial correlation reported as the effect size, the
  coefficient's p-value with `df = N - 2 - C`. It is a design decision, deliberately *not*
  part of the parameter grid.
- **which model you read** — `connectome` uses raw network strengths,
  `connectome_residualized` uses deconfounded ones. Both are computed on every run, so
  this costs nothing and needs no knob.

`edge_statistic` (including `*_partial` and `point_biserial`) and
`CPMAnalysis(calculate_residuals=...)` are deprecated onto these for one release.

### Package Structure

Source code lives in `src/cccpm/` (Poetry src layout). Tests in `tests/` with fixtures in `conftest.py` providing simulated data. The package is importable as `cccpm`.

## CI

GitHub Actions runs on push/PR to `main` and `develop`:

- **Pyflakes** (`lint` job) over `src/ tests/ examples/ scripts/`. Pyflakes only —
  undefined names, unused imports, unreachable code; no style rules. Run it locally with
  `poetry run python -m pyflakes src/ tests/ examples/ scripts/`.
- **Tests** across a matrix of ubuntu/macos/windows x Python 3.10-3.13, with coverage
  uploaded to Coveralls from the ubuntu/3.11 job.
