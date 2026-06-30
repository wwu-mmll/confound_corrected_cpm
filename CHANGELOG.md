# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
