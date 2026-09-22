# Test suite audit — 2026-09-22

A working document, not a plan. It goes test by test through all 359 and says,
for each group, **what would still fail if the code regressed**. Nothing here
has been changed; the point is for Nils to decide what goes.

Delete this file once the decisions are made — `RELEASE_PLAN.md` §2 carries the
agreed outcome.

---

## The headline number is measuring the wrong thing

Per-file wall clock, minus the ~2.8 s interpreter/import floor:

| File | Tests | Net time |
|---|---:|---:|
| `test_integration.py` | 4 | 68 s |
| `test_feature_interactions.py` | 18 | 62 s |
| `test_report_states_its_configuration.py` | 14 | 61 s |
| `test_ground_truth.py` | 17 | 41 s |
| `test_confound_api.py` | 23 | 32 s |
| `test_reporting.py` | 4 | 30 s |
| `test_models.py` | 20 | 16 s |
| `test_cpm_analysis.py` | 11 | 15 s |
| `test_no_covariates.py` | 11 | 11 s |
| **the other 15 files** | **237** | **~5 s combined** |

**237 of the 359 tests cost about five seconds in total.** Deleting them would
free no time worth having and would cost real coverage. The 336 s of runtime
lives in 122 tests, and almost all of it is `CPMAnalysis.run()` — the end-to-end
pipeline — called over and over on data whose numbers nobody checks.

Counting the actual unit of cost — `CPMAnalysis.run()`, instrumented across a
full suite run — gives **70 in-process pipeline runs** (plus three more in
subprocesses from `test_integration.py`):

| File | `run()` calls | Distinct configurations it needs |
|---|---:|---:|
| `test_feature_interactions.py` | 19 | 16 (the 2⁴ grid) + 3 |
| `test_report_states_its_configuration.py` | 17 | **5** |
| `test_ground_truth.py` | 12 | 2 (class-scoped fixtures, re-entered per test class) |
| `test_confound_api.py` | 8 | **4** |
| `test_no_covariates.py` | 6 | 5 |
| `test_cpm_analysis.py` | 4 | 4 |
| `test_models.py` | 3 | 3 |
| `test_atlases.py` | 1 | 1 |

Two rows are doing three times the work they need to. That is the runtime
problem, stated exactly.

So there are two separate questions, and conflating them is what makes the suite
feel bloated:

1. **Count.** Inflated by parametrising over things that are not test cases —
   the package's own source files, or one metric per test. Cheap to fix, saves
   no time.
2. **Time.** Driven by redundant *pipeline runs*. Fixing it barely moves the
   count.

---

## 1. Count without cost: stop parametrising over source files

| Test | Now | Should be | Lost |
|---|---:|---:|---|
| `test_module_structure.py::test_numeric_core_does_not_import_reporting` | 21 | 1 | nothing |
| `test_module_structure.py::test_numeric_core_does_not_import_plotting_stack` | 21 | 1 | nothing |
| `test_device_portability.py::test_no_gpu_only_cuda_calls` | 37 | 1 | nothing |

**−76 tests, 22% of the suite, zero coverage change.** All three are AST scans
that already build a list of offenders and name them in the assertion message;
the per-file test IDs add nothing a failure would not already tell you. Both
files stay — they are cheap and they each caught a real bug (a core module
importing the plotting stack; two bare `torch.cuda.synchronize()` calls).

---

## 2. Genuinely redundant — safe to delete

Each of these is covered *in full* by another test that does strictly more.

| Delete | Because | −n |
|---|---|---:|
| `test_models.py::TestModelInterface::test_chaining` (×4 classes) | `test_fit_predict_shape` already calls `model.fit(...).predict(...)`; chaining only adds an `isinstance` on the result | 4 |
| `test_ground_truth.py::TestEdgeSelectionStatistics` (2 tests) | `test_edge_selection.py::test_edge_selection_recovers_signed_edges` covers both signs, 2 statistics, a binary target and both `selection_input` values, in the same production path | 2 |
| `test_cpm_analysis.py::test_nan_in_X`, `test_nan_in_y` | both call `check_data` directly and assert exactly what `test_validation.py::test_missing_values_behavior` asserts; also misfiled — this file's docstring says it tests the pipeline | 2 |
| `test_scoring.py::test_mse_vs_sklearn`, `test_mae_vs_sklearn`, `test_explained_variance_vs_sklearn`, `test_pearson_vs_scipy` | all four metrics are re-checked against the same references in `test_all_metrics_vs_sklearn_random_data`, on *better* data (signal + noise rather than pure noise, so a scale error actually shows) | 4 |
| `test_ground_truth.py::test_ols_with_uniform_coefficients` | a strict special case of `test_ols_with_positive_and_negative_edges` — same assertion (`r > 0.999`), same "generative process matches the model" setup | 1 |
| `test_ground_truth.py::TestRegressionGroundTruth::test_task_type_detected`, `TestClassificationGroundTruth::test_task_type_detected` | detection is unit-tested in `test_classification.py`, and each class already asserts the *consequence* (only regression / only classification metrics in the CSV) | 2 |

**−15, nothing uncovered.**

---

## 3. Same behaviour, many test functions — merge, don't delete

The assertions stay; they move into one test. This is where the "one test per
assertion" habit shows, and the granularity is arbitrary rather than principled:
`test_validation.py` has one test making three separate claims about missing
values, and four tests each making one claim about covariate shapes.

| Group | Now | Merged | Note |
|---|---:|---:|---|
| `test_validation.py` covariate coercion (Series / 1-D / 2-D / DataFrame / bad ndim) | 5 | 1 | one table of input → expected shape |
| `test_validation.py` `get_variable_names` (3 input mixes) | 3 | 1 | one behaviour: naming fallbacks |
| `test_classification.py::TestTaskTypeDetection` | 6 | 2 | one for `detect_task_type`, one for `validate_task_type`. **Not** covered by `test_validation.py` — I checked; that file only tests `check_data` and `get_variable_names` |
| `test_inference.py` `_calculate_group_p_value` (higher/lower/≤1/NaN observed/NaN null) | 5 | 1 | one table of (observed, null) → p |
| `test_atlases.py` bundled-table assertions | 5 | 2 | one node-count/hemisphere/structure table + the whole-brain = cortical + 14 invariant. Keep the `L_precentral` landmark check inside |
| `test_atlases.py` `resolve_atlas` | 5 | 2 | one happy-path table (None / name / CSV), one error table |
| `test_simulate_sem.py` argument validation | 7 | 1 | **and add `match=`** — today they assert only that *some* `ValueError` came out, so an unrelated typo raising `ValueError` passes |
| `test_results_manager.py::test_interpretable_increments_survive` | 6 | 1 | one loop over the six metrics |
| `test_ground_truth.py` stability high/low, accuracy/AUC above chance | 4 | 2 | each pair is two halves of one claim |

**−27 collected tests, every assertion retained.**

---

## 4. Where the time actually goes — redundant pipeline runs

This is the part worth doing. None of it deletes a test.

- **`test_confound_api.py` (32 s).** `test_every_cell_of_the_2x2_runs` runs all
  four cells; `test_model_input_changes_the_connectome_model` re-runs two of
  them; `test_model_input_leaves_full_and_covariates_alone_for_the_linear_model`
  re-runs the *same* two again. Eight runs where four would do. A module-scoped
  fixture keyed on `(selection_input, model_input)` — **~15 s.**
- **`test_report_states_its_configuration.py` (61 s).** Same shape:
  `test_run_config_is_persisted` (4 cells) and
  `test_confound_control_is_explained_in_the_body` (4 cells) and
  `test_the_four_cells_produce_four_distinguishable_headlines` (4 cells) are
  twelve runs of four distinct configurations. `test_glossary_...` re-runs two
  more. **~35 s.**
- **`test_ground_truth.py` (41 s).** Already uses a class-scoped fixture, but
  pytest re-runs it per class; the regression and classification fixtures are
  each built once — this one is close to minimal. **~0 s available.**

**~50 s, about 15% of total runtime, with no change in what is asserted.**

---

## 5. Keep — these are the only cover for what they cover

- **`test_integration.py` (68 s, the most expensive file).** Runs three example
  scripts as subprocesses. This is the only thing standing between the examples
  and silent rot, and it has caught it before (a tqdm stand-in that no longer
  matched the constructor; a scalar extraction that had become a Series).
  Expensive and worth it.
- **`test_reporting.py::test_report_generates_with_atlas` (23 s).** The only
  exercise of the netplotbrain glass-brain path.
- **`test_feature_interactions.py` (62 s, 16 of its 18 tests are the 2⁴ grid).**
  The tempting cut, and the one I would not make. The file exists because twice
  a feature was inert *in combination* while working alone — that is exactly
  what a full factorial catches and a per-feature test does not. A pairwise
  covering array would drop 16 runs to ~6 and still catch pairwise
  interactions, which is what both historical bugs were. **If 40 s matters more
  than the margin, this is the honest place to take it — but it is a real
  reduction in coverage, not a free one.** My recommendation: leave it.
- **`test_statistics.py`, `test_sklearn_equivalence.py`, `test_connectome.py`.**
  Every assertion is against an external reference (statsmodels, scipy, sklearn)
  or a round-trip identity. 18 tests, ~1 s. Free and load-bearing.
- **`test_confound_api.py` design tests** (`test_selection_input_is_not_a_tuned_hyperparameter`,
  `test_nonlinear_models_are_not_invariant_to_model_input`, the OLS-invariance
  test). These encode *why* the API has the shape it has. They are the most
  valuable tests in the suite.
- **`test_simulate_simple.py` (6 tests, 0.05 s).** The simulator feeds every
  fixture in `conftest.py` and the paper's figures. Cheap insurance.

---

## 6. Has an expiry date, not redundant yet

`test_confound_api.py`'s migration guards — the 4 parametrized
`test_removed_edge_statistic_values_name_their_replacement` cases plus
`test_removed_keyword_arguments_are_gone`, `test_calculate_residuals_is_gone`,
`test_values_that_carry_over_still_work`, `test_unknown_values_are_rejected` —
pin the 0.6.x → 0.7.0 error messages. They are worth having *through* the 0.7.0
release and pointless a version later. **~8 tests to delete in 0.8.0**, noted
here so nobody has to rediscover why they exist.

---

## Totals

| | Tests | Runtime |
|---|---:|---:|
| Now | 359 | 5 m 40 s |
| §1 source-file parametrisation | −76 | — |
| §2 redundant | −15 | −2 s |
| §3 merges | −27 | — |
| §4 shared fixtures | — | −50 s |
| **After** | **241** | **~4 m 45 s** |
| §5 optional: pairwise feature grid | −10 | −40 s |

The count target is met comfortably without touching anything that carries
unique coverage. The runtime is dominated by three files that should stay
expensive.

## One thing to fix regardless of what gets cut

`test_scoring.py::test_metrics_ordering` does not test ordering. It asserts
`scores.shape[0] == N_METRICS` and then, for four metrics, that a slice has the
expected shape — true whatever order the rows are in, since indexing by the enum
is just indexing by an int in range.

The enum → row mapping *is* verified, but by
`test_all_metrics_vs_sklearn_random_data`, which indexes
`scores[Metrics.mean_squared_error]` and compares it to sklearn's MSE. So the
coverage exists; the test named for it is the one that does not provide it.
Worth noting because §2 proposes deleting four of that class's tests on the
grounds that the comprehensive one subsumes them — it does, and this is a second
reason to keep the comprehensive one rather than the singles. Either give
`test_metrics_ordering` a real ordering assertion or rename it.
