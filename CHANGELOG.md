# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
