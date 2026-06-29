# CCCPM Release Readiness Plan

Goal: make `cccpm` easy and reliable for researchers to install and use for
connectome-based predictive modeling (CPM) across macOS / Linux / Windows and
Python 3.10–3.14, with trustworthy results, modern docs, and a polished HTML report.

Status legend: [ ] todo · [~] in progress · [x] done

---

## Phase 0 — Stop the bleeding (the published package is broken)
Highest priority. A fresh `pip install cccpm` currently fails (no torch).

- [x] Add `torch` to declared dependencies (shipped in pyproject, `torch>=2.2`).
- [x] Decide torch version floor (`torch>=2.2`); platform strategy still open (Phase 3).
- [x] Reconcile version numbers: bumped 0.1.1 -> **0.3.0** (repo is now source of truth).
- [x] Add a real `LICENSE` file (MIT) at repo root.
- [x] Constrain `requires_python` to `>=3.10,<3.15`.
- [ ] Publish a fixed release to **TestPyPI** first (tag `vX.Y.Z-test`), install it
      clean on all 3 OSes, confirm `import cccpm` + run an example, THEN publish to PyPI.
      *(deferred — Nils will trigger deployment after merge to main; do NOT do this.)*

## Phase 1 — Trust the results (testing & correctness)
You don't trust the tests; turn 115 green checks into real confidence.

- [x] Reviewed coverage (63% overall). Core numerical modules are well covered
      (linear_model 99%, nonlinear 100%, scoring 99%, inner_fold 100%); the gaps are
      visualization (chord plots — Phase 5) and alternative simulation utilities.
      Added round-trip + numpy/tensor-consistency tests for the connectome <-> vector
      conversions (foundation of edge stability & p-value mapping). 8 new tests.
- [ ] Dead code: `vector_to_matrix_3d` is unused and its triu/tril index ordering
      doesn't invert `matrix_to_vector_3d` — remove it or fix+document if ever needed.
- [ ] Remaining low-coverage core paths to revisit: `edge_selection.py` (81%, the
      per-statistic branches) and `utils.py` data-insight plotting helpers.
- [ ] Audit `test_ground_truth.py`: confirm assertions check *actual numbers/edges*,
      not just "runs without error". Tighten thresholds where weak.
- [x] Added a reproducibility test (`test_pipeline_is_reproducible`): runs the full
      pipeline twice with the same seed and asserts identical aggregated metrics and
      selected edges (platform-robust; no brittle hard-coded baselines).
- [ ] Validate against an external reference (Shen et al. CPM / existing MATLAB
      results) on at least one dataset so numbers are defensible in a paper.
- [x] **Permutation p-value convention bug (statistical validity):** metric
      p-values (`_calculate_group_p_value`) and edge-FDR p-values
      (`calculate_p_values_edges_fdr`) used `(count+1)/n` instead of the correct
      `(count+1)/(n+1)` (Phipson & Smyth 2010). The old form could yield p-values
      **> 1** (when every permutation beat the observed value) and is anti-conservative.
      Fixed both; added a guard test. (The edge max-value method was already correct.)
- [ ] **Edge-selection p-value approximation (statistical validity):** the production
      path (`correlations_and_pvalues`, `point_biserial_correlation`) computes p-values
      with a normal approximation to the t-distribution. Verified vs scipy: it is
      **anti-conservative at small N** (~13% too small at N=20, ~6% at N=30, negligible
      at N>=100). The exact t-distribution can't be done on-device (`torch.special.betainc`
      absent), so route through scipy `t.sf` on CPU. *Next iteration.*
- [ ] Remaining numerical spot-checks: partial correlation (semi vs full — the function
      is named `semi_partial_*` but is validated against pingouin's full `partial_corr`),
      FDR correction, residualization.
- [x] **Input validation gap (found while writing examples):** `check_data()` now
      validates that `n_features` is a valid upper-triangular connectome size
      (`infer_n_nodes`) and raises a clear error suggesting the nearest valid sizes,
      instead of crashing deep in edge stability. Covered by new tests.
- [x] Added end-to-end edge-selection tests across all 6 statistics (pearson,
      spearman, their partials, point_biserial, point_biserial_partial). This caught
      a **real bug**: `point_biserial_partial` was completely broken — it residualized
      the binary target into continuous values and then matched it against `==0/==1`
      groups, so it selected **zero edges for every feature** (silent). Fixed by
      computing partial point-biserial as Pearson-on-residuals. Confound-controlled
      classification edge selection now works.
- [ ] Classification path: expand beyond current tests (probabilities, AUC, class
      imbalance, StratifiedKFold edge cases).
- [x] Determinism: `CPMAnalysis.__init__` no longer mutates the global NumPy/torch
      RNG. Added a `random_state` parameter; permutations now use a local
      `torch.Generator`. Verified the nonlinear models already set their own
      `random_state`, so nothing regressed. Tests assert global RNG is untouched and
      permutations are reproducible across instances (and vary with `random_state`).
- [x] Device default mismatch: `LinearCPM` and the `scoring.py` helpers defaulted to
      `device='cuda'` while `CPMAnalysis` and the nonlinear models default to `'cpu'`.
      Unified all defaults to `'cpu'` so direct use never crashes on CPU-only machines
      (the pipeline still passes `device` explicitly). Added a test.
- [ ] Still verify MPS (Apple) / CUDA paths actually run end-to-end on real hardware.
- [ ] Run the full suite on macOS (arm64), Linux, Windows locally or in CI.

## Phase 2 — Code structure / modularization
Mostly in good shape (reporting + models are already split). Targeted cleanups:

- [x] Improve top-level public API in `src/cccpm/__init__.py`: now exports
      `UnivariateEdgeSelection`, `PThreshold`, `TaskType`, models, and `__version__`.
- [x] Update stale `CLAUDE.md` key-modules table (`pytorch_model.py` ->
      `models/linear_model.py`, added nonlinear models row).
- [x] Removed the global RNG side effect (done with Phase 1 determinism).
- [~] Clean `examples/`: cruft (`tmp/`, `.ipynb_checkpoints/`, `.DS_Store`) is already
      untracked/gitignored. The two mediator examples genuinely differ and
      `example_simulated_classification.py` is now partly redundant with the
      quickstarts — **which to keep/remove is a curation call for Nils** (not deleted
      autonomously since they're author-written).
- [ ] Sanity-check heavy deps (`arakawa`, `netplotbrain`, `scikit-image`) install
      cleanly on Windows; consider making report/plot deps an optional extra.

## Phase 3 — Packaging, dependencies, cross-platform install
You explicitly care that users on Mac/Linux/Windows + various Python versions succeed.

- [ ] torch install strategy — the hard one. Default torch wheels differ per platform
      (Linux pulls large CUDA wheels; macOS arm64 = MPS; Windows = CPU). Decide:
      (a) keep default torch and document GPU separately, or
      (b) default to CPU torch + an optional `cccpm[gpu]` extra / install instructions.
- [ ] Add proper packaging metadata: `keywords`, `classifiers`, project URLs
      (homepage, docs, repository, issues), maintainers. Consider PEP 621 `[project]`.
- [ ] Set sensible version floors for numpy (1.x vs 2.x), pandas, scikit-learn,
      nilearn — verify the package works under numpy 2.x.
- [x] Expand CI test matrix: `test.yml` now runs `{ubuntu, macos, windows} ×
      {3.10, 3.11, 3.12, 3.13}` (12 jobs, `fail-fast: false`), Poetry installed via
      `pipx` for cross-OS support, coverage uploaded from one representative job.
      *Needs a real CI run (push to develop) to confirm heavy deps build on every
      combo — esp. Windows + 3.13.*
- [ ] Optional speed-up: add Poetry/pip caching (best paired with committing the lock).
- [ ] `poetry.lock` is currently **gitignored** — decide whether to commit it (helps
      CI reproducibility) and verify it installs reproducibly on all OSes.
- [x] Ran a clean-environment install smoke test locally (build wheel -> fresh venv ->
      install -> import -> tiny run). **Caught another missing-dependency bug**:
      `seaborn` (and directly-imported `scipy`, `matplotlib`, `statsmodels`) were not
      declared, so `import cccpm` failed on a clean install. All four are now declared;
      re-verified the wheel installs and runs end-to-end in a fresh venv.
- [x] Added a CI job (`.github/workflows/package_smoke.yml`) that builds the wheel,
      installs it into a clean venv (declared deps only), imports `cccpm`, and runs a
      tiny analysis — so undeclared-dependency regressions are caught automatically.

## Phase 4 — Documentation overhaul
Outdated and incomplete. Make it the on-ramp for researchers.

- [x] Fix README: corrected install, quick-start (`from cccpm import CPMAnalysis`,
      string `edge_statistic`), and repo badges.
- [x] Fix `getting_started.md`: `CPMRegression` -> `CPMAnalysis`, imports, the
      `edge_statistic` list/string bug, and the `estimate` -> `run` method name.
- [x] Rewrite `installation.md`: venv/conda quick start, per-OS tabs (macOS arm64/
      Rosetta caveat, Linux CUDA wheel size, Windows notes), CPU vs GPU/MPS guidance,
      dev install via Poetry, and a troubleshooting section (incl. the missing-torch /
      Rosetta issue). Enabled `pymdownx.tabbed`; `mkdocs build --strict` passes.
- [ ] Rewrite `getting_started.md`: replace `CPMRegression` → `CPMAnalysis`, fix
      imports and the `edge_statistic` list/string bug, verify every snippet runs.
- [x] New conceptual page `methods.md` ("How CCCPM Works") — the CPM idea, edge
      selection (incl. the statistic table), confound control (partial vs
      residualization), model variants, nested CV, edge stability, and permutation
      testing, with a recommended confound-aware workflow. Added to nav.
- [x] Fixed stale `cpm_python` badges in `index.md`.

### Paired regression + classification examples (first-class deliverable)
Every concept gets a runnable script in `examples/` **and** a matching docs tutorial,
in both flavors so users can copy the one matching their task.
- [x] `examples/regression_quickstart.py` — minimal, well-commented, simulated data.
- [x] `examples/classification_quickstart.py` — minimal, well-commented, simulated data.
- [x] Both wired into `test_integration.py` so CI verifies they run end-to-end.
- [x] Mirror both as docs tutorials: `examples/regression.md` and
      `examples/classification.md` embed the actual quickstart scripts via mkdocs
      snippets (so docs can't drift), explain each step, and cross-link to
      "Interpreting Results". Added to nav; `mkdocs build --strict` passes.
- [ ] Show the key variations in both: confound control (partial corr vs residuals),
      nested CV with p-threshold tuning, stable-edge selection, permutation testing,
      and passing `atlas_labels` for brain plots.
- [ ] Optional: a real-data (or realistic simulated) end-to-end tutorial.

### "Interpreting your results" deliverable (explain everything CCCPM outputs)
Done: `documentation/docs/interpreting_results.md` (added to nav) documents every
artifact a run produces, grounded in the actual output files, for both task types.
- [x] **HTML report pages**: Info, Data Description, Data Insights, Hyperparameters,
      Main Results, Network Strengths, Brain Plots, Edge Table — explained.
- [x] **Models & networks** explained (connectome/covariates/full/residuals/increment
      × positive/negative/both), with guidance on the `increment` model.
- [x] **Metrics**: both task types defined, with "which to trust" guidance.
- [x] **Output files**: all CSVs + npy + plots + logs documented with real columns.
- [x] Permutation p-values and edge stability explained.
- [ ] Add matching in-report captions (Phase 5, Nils-driven visual pass).
- [x] Verify API reference pages match current module names; auto-build cleanly.
      Fixed broken `api/fold.md` (`cccpm.fold` -> `cccpm.inner_fold`), `api/models.md`
      (empty package -> linear/nonlinear modules), retitled "CPM Regression" ->
      "CPM Analysis". Added a `docs` poetry group; `mkdocs build --strict` now passes.
- [x] Fix `build_docs.yml`: removed stray `photonai`, now uses Poetry + `docs` group.
- [x] Completed the `CPMAnalysis` constructor docstring (was missing `cpm_model`,
      `select_stable_edges`, `stability_threshold`, `calculate_residuals`, `device`);
      now documents every parameter and renders in the API reference.
- [x] Added class/parameter docstrings to the public `UnivariateEdgeSelection` and
      `PThreshold` classes (previously rendered nearly empty in the API reference).
- [x] Add `CHANGELOG.md` (Keep a Changelog, 0.3.0 section), `CONTRIBUTING.md`
      (dev setup, tests, docs, PR flow), and `CITATION.cff` (validated YAML, both
      contributors). *Nils: review CITATION authors/affiliations before release.*

## Phase 5 — HTML report polish (Nils-driven, visual)
You'll own the visual decisions; I can support structure + iteration speed.

- [ ] Set up a fast iteration loop: a script that regenerates the report from a fixed
      fixture and opens it, so visual changes are a few seconds to preview.
- [ ] Make plots/theming configurable; ensure the report is self-contained & portable
      (assets embedded or relative), works offline.
- [ ] Add explanatory captions so a researcher unfamiliar with the internals can read
      it; consistent terminology with the docs.
- [ ] Accessibility / print-friendliness pass.

## Phase 6 — Release
- [ ] Final version bump + CHANGELOG entry.
- [ ] Tag `-test` → TestPyPI → clean-install verification on 3 OSes.
- [ ] Tag real version → PyPI (publish.yml OIDC trusted publisher already configured).
- [ ] Post-release: install from PyPI on a clean machine, run an example, confirm docs
      site deployed.

---

## Open decisions (need Nils' input)
1. **torch/GPU packaging**: default torch + doc GPU, or CPU-default + `[gpu]` extra?
2. **Next version number**: 0.3.0 (proposed) given 0.2.1 is on PyPI?
3. **Python version ceiling**: support up to 3.13 now, or also 3.14?
4. **PEP 621 migration** for pyproject, or stay on Poetry's `[tool.poetry]` table?
5. **Reference dataset** for ground-truth validation — do we have one to validate against?
