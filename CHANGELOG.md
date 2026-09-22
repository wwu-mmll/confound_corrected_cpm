# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Confound control is now two independent run-level choices instead of four levers.**
  `UnivariateEdgeSelection` gains `selection_statistic` (`'pearson'` | `'spearman'`)
  and `selection_input` (`'raw'` | `'residualized'`); `CPMAnalysis` gains `model_input`
  (`'raw'` | `'residualized'`). `selection_input='residualized'` is the per-edge
  regression `y ~ 1 + Z + edge` — the semipartial correlation as the reported effect
  size, the coefficient's p-value with `df = N - 2 - C` — so a `p < 0.05` threshold
  means a 5% per-edge false-positive rate whatever the confounding.
  `model_input='residualized'` regresses the covariates out of the connectome the
  models consume, fitted on train and applied to test, after edge selection.
- **`Models.residuals` is removed.** Deconfounding the features is a property of the
  run (`model_input`), not a model variant. It cannot be a model name: OLS is invariant
  to it once the covariates are in the design, but the non-linear backends are not — on
  identical edges `full` moves by 391% of sd(y) for `DecisionTreeCPM`, 65% for
  `RandomForestCPM` and 19% for `GAMCPM` — so the name would mean something different
  for every backend, and users could not tell which connectome produced `full`. The
  `model` column of `cv_results_*.csv`, `cv_predictions.csv` and
  `cv_network_strengths.csv` loses the `residuals` value; run with
  `model_input='residualized'` and read `connectome` instead.
- The `presence_filter` now always sees the raw connectome. Previously
  `calculate_residuals=True` residualised the connectome before selection, so the
  filter looked for structural zeros in residualised values, where they no longer
  exist. That interaction warning is gone with the cause.

### Removed
No deprecation shims: this release changes what some parameters *mean*, and code
that keeps running while quietly producing different numbers is worse than code that
stops. Every removal raises with its replacement named.

- `UnivariateEdgeSelection(edge_statistic=...)`. `'pearson'` and `'spearman'` carry
  over unchanged as `selection_statistic`; the rest map as
  `'point_biserial'` -> `selection_statistic='pearson'`,
  `'pearson_partial'`/`'spearman_partial'`/`'point_biserial_partial'` -> the same
  statistic with `selection_input='residualized'`. `'point_biserial'` was never a
  separate statistic — Pearson against a 0/1 target *is* the point-biserial
  correlation.
- `CPMAnalysis(calculate_residuals=...)`. Use `selection_input='residualized'` with
  `model_input='residualized'`. **Edge sets change**: the old path selected with a
  plain correlation on residualised edges, whose effective alpha shrank as
  confounding grew (measured 5.1% -> 3.3% -> 2.1% at nominal 5%); the new one is
  nominal at every confound level.
- `Models.residuals`, as described above.

- **`increment` is no longer reported for metrics whose difference is not a
  statistic.** It is a difference of two metrics, which only means something where
  differences of that quantity are standard: explained variance, the error metrics
  (as error reduction), accuracy, balanced accuracy and ROC AUC. It is now NaN for
  **Pearson r** — comparing two correlations needs Fisher z or Steiger's test, not
  subtraction — and for **F1**, a harmonic mean whose difference has no established
  interpretation. See `constants.INCREMENTABLE_METRICS`.
- Report tables render a deliberate NaN as an em dash rather than the string `nan`,
  and the Model Comparison section explains why `increment` shows one for Pearson r
  and F1, so it does not read as a failed computation.
- The Analysis Configuration table no longer turns the wrapped lines of a
  multi-line estimator repr into their own empty rows.

### Added
- **The report states which confound configuration produced it.** `run_config.json`
  records `selection_input` / `model_input` / `selection_statistic` alongside
  `task_type.txt`; the headline and a stat chip name the cell of the 2x2
  (`none` / `edge selection only` / `features only` / `edge selection and features`),
  and the Model Comparison section explains what it means. Previously a naive run and
  a fully controlled one opened with the identical sentence and the configuration was
  only in the appendix table. Result directories written before 0.7.0 say nothing
  rather than guessing.
- The model glossary describes the models *this* run produced — under
  `model_input='residualized'`, `connectome` is labelled as the deconfounded-strength
  model rather than "connectivity alone".
- Vanilla CPM without covariates: `covariates` is optional in `CPMAnalysis.run`. Only
  the `connectome` model is defined; the rest are NaN and `available_models.json`
  records which rows are real. Options requiring covariates raise up front.
- `pyflakes` runs in CI.

### Fixed
- Permutation p-values for undefined models were reported at the permutation floor
  (`1/(n_perms+1)`, i.e. maximally significant) instead of NaN.
- Classification metrics turned NaN predictions into a plausible-looking score near the
  base rate, because every comparison against NaN is silently False.
- `simulate_confounded_data_chyzhyk` raised `UnboundLocalError` for an invalid
  `link_type` instead of a `ValueError` naming the valid options.

## [0.5.0] — 2026-08-26

### Added
- **Connected-component edge selection** (`UnivariateEdgeSelection(connected_components=...)`).
  Optionally keeps only selected edges that belong to a connected component (per
  positive/negative network) with at least a minimum number of edges, dropping isolated
  single edges to favour coherent subnetworks and improve stability. `True` drops lone
  edges (min 2 edges); an int sets the minimum. Applied per fold and per permutation.
- **Brain-plot edge thresholding + in-report selector.** The Brain & Edges section now
  defaults to the *significantly stable* edges (NBS/TFCE p < 0.05) instead of all stable
  edges, and offers buttons to switch the connectivity matrix, hub, network-summary and
  chord views between **Significant**, **Top 5%** and **Top 10%** (by stability). Without
  permutation-based significance the default is all stable edges. The glass brain renders
  the default subset. Fully self-contained (inline JS/CSS, no external requests).
- **Presence filter for edge selection** (`UnivariateEdgeSelection(presence_filter=...)`).
  Optionally keeps only edges that are nonzero in at least a given fraction of subjects
  (`True` = majority/0.5, or a float), dropping structural/near-zero edges before
  selection. Intended for sparse structural connectomes (e.g. DTI streamline counts);
  computed per fold on the training subjects from the connectome only, so it adds no
  target leakage, and is additive to the existing near-zero-variance gate. Leave off
  (default `False`) for functional data whose edges vary around a mean of ~0.

### Fixed
- **GPU out-of-memory in edge-stability aggregation.** `ResultsManager` preallocated the
  per-fold edge masks `[Features, 2, Folds, Runs]` on the compute device and densified
  them to node×node arrays there, so large parcellations × many folds × many permutations
  could need tens of GB of VRAM (e.g. ~25 GB just for `cv_edges` at 500 nodes × 100 folds
  × 1000 permutations, before a ~200 GB `edges.npy` densification). Edge bookkeeping now
  lives on the CPU and is reduced to a running fold-sum (all that stability needs), the
  node×node arrays are built on the CPU, and per-fold `edges.npy` is written for the real
  run only — the permutation pass keeps just the fold-averaged `stability_edges.npy` it
  needs for the null. Peak VRAM for this step is now negligible. See the installation
  guide for running big analyses on limited-VRAM GPUs.
- **RepeatedKFold individual-level outputs.** Predictions and network strengths are now
  tagged with a `repeat` id (and network strengths with a `sample_index`), and the HTML
  report averages each subject's values across the repeats of a `RepeatedKFold` before
  plotting. Previously every subject appeared `n_repeats` times in the predicted-vs-observed
  scatter and its pooled-correlation annotation. Edge stability and the fold-level metric
  summary are unchanged — they still pool all folds and repeats.
- **GPU device mismatch in confound-controlled edge selection.** `get_residuals`
  built the intercept column on the CPU, so the `*_partial` edge statistics crashed
  with `Expected all tensors to be on the same device` when running on `device='cuda'`.
  The intercept is now created on the inputs' device, and the numpy-return path moves
  through `.cpu()` first. Added a CUDA-guarded regression test.

### Deprecated
- The never-functional `t_test_filter` argument is replaced by `presence_filter`.
  Passing `t_test_filter` now warns and is ignored.

### Removed
- Dead code cleanup (no behavior change): the broken/unused `vector_to_matrix_3d`,
  `matrix_to_upper_triangular_vector`, and `vector_to_upper_triangular_matrix` helpers and a
  duplicate `import`; the uncallable `ResultsManager.collect_results` (referenced undefined
  names) and the unused `_save_inner_cv_to_csv`; the non-functional `SelectPercentile` /
  `SelectKBest` edge-selector stubs; and the unused `simulation/simulate_multivariate`
  module. `cv_predictions.csv` is now written once (was written twice per run).

## [0.4.1] — 2026-07-02

Correctness fixes and dead-code cleanup.

### Fixed
- **Predictive increment baseline.** The `increment` model now computes
  `full − covariates` (the connectome's added value over the confounds), matching the
  report and documentation, instead of `full − connectome`. The previous formula measured
  the covariates' increment over the connectome and read ≈ 0 even when the `residuals`
  model showed genuine confound-independent brain signal.
- **TFCE threshold-sweep rounding.** Added a tolerance to the height comparison in the
  network TFCE statistic so an edge whose stability lands exactly on a sweep gridpoint
  reliably contributes at its own height. Accumulated `numpy.arange` rounding (e.g.
  `0.8 → 0.8000000000000001`) could otherwise drop an edge's top contribution
  non-deterministically across numpy versions.

### Removed
- Dead code: the uncallable `calculate_final_cv_results_old` method (referenced an
  undefined name), the unused `chord_v2` plotting module, and unused imports.

## [0.4.0] — 2026-07-01

Subnetwork-level significance for edge stability.

### Added
- **Network-Based Statistic (NBS) for edge-stability significance** (now the default,
  `edge_significance_method="nbs"`). Edges whose stability meets `nbs_threshold` form a
  graph; its connected components are tested against a permutation null of the largest
  component (`nbs_component_stat="extent"` or `"intensity"`), controlling the family-wise
  error rate at the subnetwork level. New `CPMAnalysis` parameters:
  `edge_significance_method`, `nbs_threshold`, `nbs_component_stat`.
- **Network TFCE** (`edge_significance_method="tfce"`): Threshold-Free Cluster Enhancement
  adapted to networks, giving per-edge FWER-corrected p-values with no primary threshold.
- **Significance diagnostics** persisted to `stability_edges_significance_meta.json`
  (method, parameters, the permutation null distribution, and — for NBS — the observed
  components and largest-component size).
- **Report — Stable Edges section**: names the significance method and its parameters,
  reports per-network diagnostics (largest component, # significant), renders a permutation
  null-distribution plot per network, shows **all** significant edges (no cap), and offers a
  downloadable CSV of every selected edge with its stability and significance.
- Documentation of edge/network significance in "How CCCPM Works", "Getting Started", and
  "Interpreting Results".

### Removed
- The previous per-edge significance methods (Benjamini–Yekutieli FDR and the per-edge
  max-statistic), which were underpowered for connectome-scale edge counts — FDR needs raw
  p-values below the `1/(n_perm+1)` resolution floor, and the max-statistic is crippled by
  the discreteness of stability. Superseded by NBS/TFCE above.

## [0.3.2] — 2026-07-01

Documentation and examples: SEM-based simulated data.

### Added
- **New "Simulating Data" documentation page** covering the SEM-based simulator
  (`cccpm.simulation.simulate_sem`): the common-cause generative model, the four
  interpretable edge classes (mixed / pure-signal / confound-only / noise), the
  `R2` and `kappa` parameterisations, `generate_confound_grid`, and how to binarise
  the target for classification. Added a matching mkdocstrings API reference page.

### Changed
- **Migrated the example scripts to the SEM-based simulator.**
  `regression_quickstart.py`, `classification_quickstart.py` (median-split of the
  continuous outcome), and `example_simulated_data.py` now generate confound-aware
  data with a known ground-truth R² via `simulate_data_given_kappa`, instead of the
  older `simulate_simple` generator.

### Removed
- Redundant example scripts (`example_simulated_classification.py`, the two
  `mediator_*` examples) and the unused `simulation/mediator_simulation.py` module.

## [0.3.1] — 2026-06-30

HTML report redesign.

### Changed
- **Rebuilt the HTML report on Jinja2 + CSS instead of `arakawa`.** The report is now a
  single, self-contained, offline page (figures embedded as inline SVG / base64) with a
  sticky-sidebar table of contents and a print stylesheet. `arakawa` (a ~1.7 MB React
  bundle) is no longer a dependency.
- **Redesigned the layout into a top-down narrative**: a **Summary** with a one-sentence
  verdict, key-stat chips, and predicted-vs-observed scatter plots for the positive /
  negative / both networks plus the covariates baseline (each annotated with its
  cross-validated effect size and permutation *p*); **Model Comparison** (one faceted
  figure + the APA results table); **Network Strengths**; **Brain & Edges**; **Stable
  Edges**; and a **Data & Methods** appendix. Every section has an always-visible
  explanatory note, and a design-token-based stylesheet for a consistent look.
- All report figures are generated at standardized sizes and saved as vector SVG.

### Added
- **Neuroscience figures** (`cccpm.reporting.plots.brain_figures`): connectivity matrix,
  network-summary matrix, chord diagram (via `pycirclize`), node-degree/hub plot, and a
  glass-brain rendering of the stable edges. The matrix and hub plots need no atlas; the
  network-summary and chord need a `network` column; the glass brain needs node
  coordinates. New `pycirclize` dependency.
- `scripts/preview_report.py` to regenerate the report from a fixture for fast iteration,
  and reporting smoke tests.

### Fixed
- The glass-brain figure no longer silently disappears (it was loading
  `sig_stability_*` files the pipeline does not write); it is now built from the
  stability matrices.
- The effect size annotated on the summary scatter now matches the hero verdict
  (cross-validated mean, not the pooled-points correlation).

## [0.3.0] — 2026-06-30

Release-readiness pass focused on a reliable install, cross-platform support, and
documentation.

### Fixed
- **`pip install cccpm` now works**: `torch` is declared as a dependency. Previous
  releases imported `torch` everywhere but never required it, so a fresh install
  failed at import time.
- **Declared previously-undeclared runtime dependencies** (`scipy`, `seaborn`,
  `matplotlib`, `statsmodels`). They were imported directly but only present
  transitively, so a clean `pip install cccpm` could fail at import (e.g.
  `ModuleNotFoundError: No module named 'seaborn'`).
- **`point_biserial_partial` edge selection was broken** and silently selected no
  edges (it residualized the binary target into continuous values, then compared it
  against the `0`/`1` groups). Partial point-biserial is now computed as Pearson on
  the residuals, so confound-controlled edge selection works for classification.
- **Point-biserial correlation was inflated.** The binary-target edge statistic used
  the pooled within-group standard deviation as its denominator instead of the total
  standard deviation of the feature, overstating `|r|` (with imbalanced groups it could
  saturate at 1.0). It now equals `scipy.stats.pointbiserialr`.
- **Permutation p-values used an invalid denominator.** Metric and edge-FDR
  permutation p-values were computed as `(count + 1) / n_permutations`, which is
  anti-conservative and could even exceed 1. They now use the standard
  `(count + 1) / (n_permutations + 1)` (Phipson & Smyth, 2010).
- Unified all `device` defaults to `"cpu"` (`LinearCPM` and the `scoring` helpers
  previously defaulted to `"cuda"`), so direct use never crashes on machines
  without a GPU. The pipeline still uses the GPU when requested.
- `check_data()` now fails fast with a clear message when `X` does not have a valid
  connectome size (`n_features = n_nodes * (n_nodes - 1) / 2`), instead of crashing
  deep in edge-stability computation.
- Fixed the documentation build (`mkdocs build --strict` now passes) and corrected
  outdated API references and code examples.

### Added
- Top-level public API: `CPMAnalysis`, `UnivariateEdgeSelection`, `PThreshold`,
  `TaskType`, and the model classes are now importable directly from `cccpm`, along
  with `cccpm.__version__`.
- Paired, runnable quickstart examples for regression and classification
  (`examples/regression_quickstart.py`, `examples/classification_quickstart.py`),
  verified end-to-end in CI.
- Documentation: an "Interpreting Results" guide, regression/classification example
  tutorials, and a rewritten cross-platform installation guide.
- Cross-platform CI test matrix: `{ubuntu, macos, windows} × {3.10–3.13}`.
- Project files: `LICENSE` (MIT), `CITATION.cff`, and `CONTRIBUTING.md`.

### Changed
- `CPMAnalysis` no longer reseeds the global NumPy/torch RNG on construction. It
  now takes a `random_state` parameter (default 42) and uses a local generator for
  permutations, so creating a `CPMAnalysis` no longer affects other code in your
  script. Results remain reproducible.
- Require Python `>=3.10,<3.15` and `torch>=2.2`.
- Added packaging metadata (keywords, classifiers, project URLs).
- Unified edge selection into a single vectorised OLS/GLM path (Pearson, Spearman,
  their confound-controlled variants, and point-biserial for binary targets all share
  one residualise-then-correlate implementation). When controlling for confounds, the
  reported `r` is the semi-partial correlation (confounds removed from the connectome
  edge, not the target); selection p-values are unchanged by this choice.

---

Earlier releases (0.2.x and prior) predate this changelog; see the git history for
details.
