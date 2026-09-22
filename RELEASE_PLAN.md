# CCCPM Release Plan — what is left to do

Goal: ship `cccpm` 0.7.0 — easy and reliable to install and use for
connectome-based predictive modeling (CPM) across macOS / Linux / Windows and
Python 3.10–3.14, with trustworthy results, modern docs, and a polished HTML
report.

**Last audited: 2026-09-22** against `0.7.0` (358 tests green locally, CI red).
This file lists **only open work**. Finished work is not recorded here — the
record lives in git history and `CHANGELOG.md`. Items are ordered
most-important-first, both between and within sections.

Status legend: `[ ]` todo · `[~]` in progress

---

## 0. Blocking the 0.7.0 release

In order:

1. **Cut the release** — `CHANGELOG.md`'s `[Unreleased]` heading becomes
   `[0.7.0]` with a date (`pyproject.toml` already says 0.7.0), then §4.

   CI is green and the docs now teach the 0.7.0 API, so the two things that were
   blocking are done. What remains before tagging is §4: the packaging
   decisions, and a TestPyPI dry-run with a clean install on all three OSes.

---

## 1. CI

Green as of 2026-09-22 — all 13 jobs on `c5bd54b`, plus the package smoke
test. The two failures
were, for the record: `package_smoke.yml` still passing `edge_statistic=`
(removed in `4c1ffb4` along with the deprecation shims), and a base64 lottery
in `tests/test_no_covariates.py` — its `_rendered_text` stripped only
`data:image/...;base64,` followed *immediately* by base64, missing matplotlib's
SVG payloads, which write a newline after the comma. ~57 KB of base64 reached a
`"nan" not in ...` substring assertion; expected hits by chance 1.75 per report.
The bytes differ per OS because the rendered figures do, which is why the
failure split cleanly by operating system rather than by Python version.

- [ ] **Production code still reads and writes files with the locale codec.**
      `reporting/reporting_utils.py:129`, `reporting/data_loader.py:83,98,220,250`,
      `results_manager.py:363`, `cpm_analysis.py:379,387`, `inference.py:144`.
      Symmetric on one machine, so it passes CI, but a report written on a
      cp1252 Windows box is not readable elsewhere and a non-cp1252 character
      in a target name raises on write. The tests now all pass `encoding=`;
      the package should too.

## 2. Test suite: 287 tests, 4m27s local

Down from 359 / 5m40s. The source-file parametrisations are collapsed (two AST
checks that were 79 collected tests are now 3, each verified to still name the
offending file and line) and the repeated pipeline runs are shared behind
module-scoped fixtures: `test_confound_api.py` 34.7s → 19.8s,
`test_report_states_its_configuration.py` 63.5s → 25.1s.

Worth keeping in mind for anything further: the count and the runtime are nearly
disjoint problems. 237 of the tests cost about five seconds combined — the time
is repeated `CPMAnalysis.run()` calls, not the number of test functions.

**Awaiting Nils' decision.** Each of these was checked against what would still
fail if the code regressed; none is the only cover for anything.

- [ ] **Delete the 15 redundant tests**, each covered in full by a test that
      does strictly more:
      `test_models.py::TestModelInterface::test_chaining` (×4 model classes —
      `test_fit_predict_shape` already calls `fit(...).predict(...)`);
      `test_ground_truth.py::TestEdgeSelectionStatistics` (2 —
      `test_edge_selection.py::test_edge_selection_recovers_signed_edges` covers
      both signs, 2 statistics, a binary target and both `selection_input`
      values); `test_cpm_analysis.py::test_nan_in_X` and `test_nan_in_y` (2 —
      identical to `test_validation.py::test_missing_values_behavior`, and
      misfiled); `test_scoring.py`'s `test_mse_vs_sklearn` /
      `test_mae_vs_sklearn` / `test_explained_variance_vs_sklearn` /
      `test_pearson_vs_scipy` (4 — subsumed by
      `test_all_metrics_vs_sklearn_random_data`, which uses signal + noise
      rather than pure noise so a scale error actually shows);
      `test_ground_truth.py::test_ols_with_uniform_coefficients` (1 — a strict
      special case of `test_ols_with_positive_and_negative_edges`); the two
      `test_task_type_detected` cases (2 — unit-tested in
      `test_classification.py`, and each class already asserts the consequence).
- [ ] **Merge 27 collected tests into 9**, retaining every assertion, where one
      behaviour is spread over many test functions: `test_validation.py`'s
      covariate coercion (5→1) and `get_variable_names` (3→1);
      `test_classification.py::TestTaskTypeDetection` (6→2 — *not* covered by
      `test_validation.py`, which only tests `check_data` and
      `get_variable_names`); `test_inference.py`'s `_calculate_group_p_value`
      cases (5→1); `test_atlases.py`'s bundled-table assertions (5→2) and
      `resolve_atlas` (5→2); `test_simulate_sem.py`'s argument validation (7→1,
      **and add `match=`** — today they assert only that *some* `ValueError`
      came out); `test_results_manager.py::test_interpretable_increments_survive`
      (6→1).
- [ ] **Delete the migration guards in 0.8.0**, not now: the ~8 tests in
      `test_confound_api.py` pinning the removed-0.6.x-spelling errors, plus the
      `test_renamed_parameters_*` cases added for
      `stability_significance_method` / `nbs_stability_threshold`. Worth having
      through this release, pointless after it.
- [ ] `test_scoring.py::test_metrics_ordering` asserts shapes, not ordering —
      true for any in-range index. The enum-to-row mapping *is* checked, but by
      `test_all_metrics_vs_sklearn_random_data`. Give it a real assertion or
      rename it.

Leave the three expensive files alone: `test_integration.py` (the only guard
against example rot), `test_reporting.py`'s atlas test (the only netplotbrain
cover) and `test_feature_interactions.py`'s 2^4 grid (62s — it exists because
two features were silently inert *in combination*, which is what a full
factorial catches and per-feature tests do not; a pairwise covering array would
save 40s but is a real reduction in coverage).

## 3. Docs & report

The 0.6.x API references are gone: `methods.md`, `getting_started.md`,
`interpreting_results.md`, both example pages, the package `README.md`, the
quickstarts and the embedded showcase report were all rewritten against 0.7.0,
and `mkdocs build --strict` is clean. `api/statistics.md` and `api/inference.md`
were added — the two modules split out in 0.7.0 had no API reference at all.

- [ ] Add in-report captions; finish the accessibility/print audit.
- [ ] Optional: a real-data (or realistic simulated) end-to-end tutorial.
- [ ] Regenerate `documentation/docs/assets/simulated_data_report.html` whenever
      the report layout changes — it is a committed 2.2 MB artifact and will go
      stale silently. Generated by a snippet kept with the release notes;
      consider a `scripts/` entry point so it is reproducible.

## 4. Packaging & cross-platform install

- [ ] **torch install strategy** (decision #1): keep default torch + document
      GPU, or CPU-default + a `cccpm[gpu]` extra.
- [ ] Set version floors for numpy (1.x vs 2.x), pandas, scikit-learn, nilearn;
      verify under numpy 2.x.
- [ ] Decide whether to commit `poetry.lock` (CI reproducibility) and whether
      to migrate pyproject to PEP 621 (decision #4). Poetry 2.x now warns on
      every `[tool.poetry]` metadata key, so this is no longer cosmetic.
- [ ] Sanity-check heavy report deps (`netplotbrain`, `scikit-image`,
      `pycirclize`) on Windows; consider an optional `cccpm[plots]` extra.
- [ ] **matplotlib backend (Windows):** rewrite report plotting to the OO
      `Figure()` API so it renders via Agg regardless of the user's backend,
      without a global `matplotlib.use("Agg")` (workaround documented in
      `installation.md`: `MPLBACKEND=Agg`).
- [ ] **Release:** `vX.Y.Z-test` → TestPyPI dry-run, clean-install on all 3
      OSes, then tag for PyPI. (Nils triggers deployment.)
- [ ] **`poetry install --with docs` takes >10 minutes.** The docs group is not
      in the local lock, so it forces a full re-resolution of a dependency set
      that includes torch, nilearn and netplotbrain. Sharpens the `poetry.lock`
      decision above: CI's `build_docs.yml` pays this on every push to `main`.

## 5. Correctness & statistical validity

- [~] **Edge-selection p-value approximation** (decision #6). The normal
      approximation to the t-tail in `correlations_and_pvalues` is
      anti-conservative at small N. Now measured, not estimated:
      `test_sklearn_equivalence.py::test_edge_pvalues_vs_exact_t_distribution`
      pins max |p_exact − p_approx| at 0.018 (n=30), 0.0089 (n=60), 0.0044
      (n=120), 0.0010 (n=500), and asserts the error is never conservative.
      Pick (a) keep, (b) on-GPU exact t-tail, or (c) scipy `t.sf` on CPU for
      the threshold step only. Needs sign-off; whichever is chosen, that test
      says what changes.
- [ ] **External reference validation** vs Shen MATLAB / GenCPM on ≥1 dataset,
      so paper numbers are defensible. This is what the `cccpm_paper` benchmark
      is for — feed the result back here once numbers agree. See
      `../PAPER_PLAN.md` §3.
- [ ] **Classification path:** expand tests (probabilities, AUC, class
      imbalance, StratifiedKFold edge cases). Coordinate with §2 — this adds
      tests while §2 removes them; both are about coverage, not count.
- [ ] **Verify MPS (Apple) / CUDA** run end-to-end on real hardware.
- [ ] `cpm_analysis.py:474` — `torch.as_tensor` on a read-only,
      DataFrame-derived array shares memory instead of copying. Safe today,
      latent hazard, still emits a `UserWarning`. Deserves its own commit.

## 6. Code health

- [ ] `src/cccpm/reporting/plots/cpm_chord_plot.py` had a `__main__` block
      hardcoded to a `/spm-data` server path, shipped in the wheel — removed.
      Worth a sweep for others like it: development entry points inside the
      package are invisible to pyflakes and to the tests.

---

## Open decisions (need Nils' input)

1. **torch/GPU packaging**: default torch + doc GPU, or CPU-default + `[gpu]`
   extra?
4. **PEP 621 migration** for pyproject, or stay on Poetry's `[tool.poetry]`
   table? (Poetry 2.2 warns on every key now.)
6. **Edge-selection p-value computation**: keep the GPU normal approximation
   (a), a hand-rolled on-GPU exact t-tail (b; `torch.special.betainc` is
   missing in torch 2.x), or scipy `t.sf` on CPU for the p-value step only (c).
   Recommend (b) to stay GPU-pure, else (c).
7. **Multiple-comparison / FDR in edge selection**: `PThreshold` supports
   statsmodels corrections (default `None`, no bug). Should CPM correct across
   the ~tens-of-thousands of edges by default, and with which method? CPM is
   traditionally run uncorrected at a liberal threshold because the model +
   permutation test provide the real inferential control — decide deliberately.
   (Distinct from edge-*stability* significance, resolved via NBS/TFCE.)
8. **`balanced_accuracy` in `INCREMENTABLE_METRICS`** — kept because it is a
   mean of two proportions; the written plan listed only `accuracy`/`roc_auc`.
   Confirm or revert.
9. **Should one run emit all four 2×2 cells?** Both axes are run-level now, so
   this needs axes over both — not free. Currently four runs.

## Logged, deliberately not being built now

- **Cross-run comparison report.** Confound control is two run-level choices,
  so the 2×2 is four separate analyses and no single report can show the
  raw-vs-deconfounded comparison that is the scientifically interesting output.
  A report that reads several results directories and puts them side by side
  would. `cccpm_paper`'s `figures/fig_empirical.py` does this by hand today.
  Agreed with Nils to log it rather than build it — it is a new feature, not
  reporting follow-through.
- **Extract the brain figures into `wwu-mmll/brainplots`**, publish it, and
  depend on it via `cccpm[plots]` (code currently lives in CCCPM).
- **Make the glass brain honour the Significant / Top 5% / Top 10% selector**
  (the matrix/hub/chord views already do). Needs netplotbrain render cost
  addressed first.

---

*Released: 0.3.0 (install reliability), 0.3.1 (HTML report redesign), 0.4.0
(NBS/TFCE edge-stability significance), 0.4.1 (increment baseline + TFCE
fixes), 0.4.2 (built-in atlas registry), 0.5.0.*
