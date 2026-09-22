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

1. **CI is red** — see §1. A release cut from a red pipeline is a release nobody
   can verify.
2. **The mkdocs documentation still teaches the 0.6.x API** — see §3.
3. **Cut the release** — `CHANGELOG.md`'s `[Unreleased]` heading becomes
   `[0.7.0]` with a date (`pyproject.toml` already says 0.7.0), then §4.

---

## 1. CI

Fixed 2026-09-22, **not yet confirmed green on the runners**. The two failures
were, for the record: `package_smoke.yml` still passing `edge_statistic=`
(removed in `4c1ffb4` along with the deprecation shims), and a base64 lottery
in `tests/test_no_covariates.py` — its `_rendered_text` stripped only
`data:image/...;base64,` followed *immediately* by base64, missing matplotlib's
SVG payloads, which write a newline after the comma. ~57 KB of base64 reached a
`"nan" not in ...` substring assertion; expected hits by chance 1.75 per report.
The bytes differ per OS because the rendered figures do, which is why the
failure split cleanly by operating system rather than by Python version.

- [ ] **Confirm green on all 13 jobs** after pushing, before tagging.
- [ ] **Production code still reads and writes files with the locale codec.**
      `reporting/reporting_utils.py:129`, `reporting/data_loader.py:83,98,220,250`,
      `results_manager.py:363`, `cpm_analysis.py:379,387`, `inference.py:144`.
      Symmetric on one machine, so it passes CI, but a report written on a
      cp1252 Windows box is not readable elsewhere and a non-cp1252 character
      in a target name raises on write. The tests now all pass `encoding=`;
      the package should too.

## 2. Test suite: 358 tests, 5m38s local (~13 min on the Windows runner)

The suite grew past the point where its size is itself a maintenance cost.
Target **~250 tests** and **~4 min** local, with no loss of real coverage.
Principle: *one test per behaviour, not per assertion; never parametrize over
source files.*

- [ ] **Stop parametrizing over the package's own files (−76 tests).**
      `test_module_structure.py` is two AST checks × 21 core modules = 42 tests;
      `test_device_portability.py` is one AST check × 37 files = 37 tests. Each
      check should be a single test that walks the files and collects offenders
      — the assertion messages already name the offending module, so the
      per-file test IDs buy nothing. 79 tests → 3.
- [ ] **Share the four-cell CPM runs in `test_report_states_its_configuration.py`**
      (14 tests, ~40s). `test_run_config_is_persisted` and
      `test_confound_control_is_explained_in_the_body` each re-run a full
      analysis for all four 2×2 cells at ~3.8s apiece. A module-scoped fixture
      that runs the four cells once cuts ~30s and lets the four-headline test
      reuse them too.
- [ ] **Collapse one-assertion families.** Argument-validation tests in
      `test_validation.py` (15), `test_simulate_sem.py` (12),
      `test_simulate_simple.py` (6) and `test_atlases.py` (18) are largely one
      `pytest.raises` or one shape check each; a single table-driven test per
      family says the same thing. Roughly −19.
- [ ] **Remove genuine duplication.** `test_scoring.py`'s
      `test_mse_vs_sklearn` / `test_mae_vs_sklearn` /
      `test_explained_variance_vs_sklearn` / `test_pearson_vs_scipy` are
      subsumed by `test_all_metrics_vs_sklearn_random_data` in the same class
      (−4). `TestTaskTypeDetection` in `test_classification.py` overlaps
      `test_validation.py` (−3). The three
      `test_calculate_group_p_value_*` cases in `test_inference.py` are one
      behaviour (−2).
- [ ] **Leave the slow tests alone.** `test_integration.py` (69s: three example
      scripts as subprocesses) and `test_reporting.py::test_report_generates_with_atlas`
      (23s: the netplotbrain path) are the most expensive tests in the suite and
      also the only ones covering what they cover. They stay.
- [ ] Each deletion must be justified by *what would still fail* if the code
      regressed. Anything that is the only cover for a behaviour stays,
      however small.

## 3. Docs & report

- [ ] **The mkdocs documentation still describes the 0.6.x API** (found
      2026-09-22). `documentation/docs/methods.md` (8 references) and
      `getting_started.md` (3) use `edge_statistic=`, `calculate_residuals=`
      and the `residuals` model, all removed in 0.7.0 — anyone following them
      gets a `TypeError` or a `ValueError`. Replace with `selection_statistic`
      / `selection_input` / `model_input`, and rewrite the model list
      (`connectome` / `covariates` / `full` / `increment`, with `increment`
      NaN for Pearson r and F1). **Blocks the release.**
- [ ] Show key variations in both quickstarts: confound control
      (`selection_input` vs `model_input`), nested CV with p-threshold tuning,
      stable-edge selection, permutation testing, and passing `atlas` for brain
      plots.
- [ ] Add in-report captions; finish the accessibility/print audit.
- [ ] Optional: a real-data (or realistic simulated) end-to-end tutorial.
- [~] `examples/` curation: `example_simulated_classification.py` overlaps the
      quickstarts — which author-written examples to keep is a call for Nils.

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

No open items.

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
