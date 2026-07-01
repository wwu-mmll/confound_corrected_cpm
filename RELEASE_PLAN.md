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
- [x] **Independent numerical validation vs scikit-learn / scipy**
      (`tests/test_sklearn_equivalence.py`): the vectorised torch/CUDA edge statistics
      (Pearson, partial/semipartial, Spearman) and the batched OLS solvers for all four
      CPM model variants (connectome/covariates/full/residuals) match scipy/numpy/sklearn
      references (effect sizes to 1e-6, predictions to 2e-3, identical selected-edge masks);
      the full CV pipeline reproduces an independent sklearn pipeline. Also exercises the
      previously-thin `edge_selection.py` branches.
- [x] **Permutation p-value convention bug (statistical validity):** metric
      p-values (`_calculate_group_p_value`) and edge-FDR p-values
      (`calculate_p_values_edges_fdr`) used `(count+1)/n` instead of the correct
      `(count+1)/(n+1)` (Phipson & Smyth 2010). The old form could yield p-values
      **> 1** (when every permutation beat the observed value) and is anti-conservative.
      Fixed both; added a guard test. (The edge max-value method was already correct.)
- [~] **Edge-selection p-value approximation (statistical validity) — DECISION PENDING
      (see Open decisions #6).** The production path (`correlations_and_pvalues`,
      `point_biserial_correlation`) computes p-values with a normal approximation to the
      t-distribution (lines ~219-222, 268-273, 345-348 in `edge_selection.py`). Verified
      vs scipy: **anti-conservative at small N** (~13% too low at N=20, ~6% at N=30,
      negligible at N>=100). Nils wants to keep a torch/GPU-friendly path, so we will NOT
      route through scipy yet — decide later between (a) keep the current approximation,
      (b) a more accurate on-GPU approximation, or (c) exact via scipy on CPU.
- Remaining numerical spot-checks: partial correlation (semi vs full), point-biserial,
  FDR correction, residualization. Broken out:
  - [x] **Point-biserial base formula bug (statistical validity):** the non-partial
        `point_biserial_correlation` used the **pooled within-group SD** as the
        denominator of `(M1-M0)/s * sqrt(n0 n1/(n0+n1)^2)` instead of the **total SD
        of X**. This inflates |r|; with imbalanced groups + strong separation it drove
        r to the clamp (1.0) where `scipy.stats.pointbiserialr` gives ~0.78. Replaced
        with Pearson-on-standardized X/Y (point-biserial *is* Pearson against a 0/1
        target) — now matches scipy to ~1e-8. Added a parametrized regression test.
  - [x] **Partial vs semi-partial — RESOLVED via unified OLS GLM (discussed w/ Nils).**
        The old `*_partial` path computed a **full partial** correlation (both X and Y
        residualized). Decided: edge selection is one **OLS GLM** — `target ~ intercept
        [+ confounds] + edge`, testing the edge coefficient. By Frisch–Waugh–Lovell this
        only residualizes the **edge** (the target is never residualized for the
        coefficient — important since y is the ML prediction target). The reported `r`
        is now the **semi-partial** correlation (confound removed from the connectome
        only); its sign and the p-value are the regression coefficient's, so the
        partial-vs-semipartial choice does **not** change which edges are selected, only
        the reported effect-size magnitude. Validated p vs statsmodels OLS coefficient
        test and r vs the semi-partial definition. 35k edges × 1000 perms = 0.25s (CPU).
  - [x] **Unified edge-selection path (code reduction).** Collapsed the separate
        `point_biserial_correlation` + per-statistic branches into one vectorised
        `correlations_and_pvalues` (residualize-once + matmul, batched over all perms).
        A binary 0/1 target flows through OLS with no special-casing (= point-biserial /
        linear-probability model). Spearman = same path on ranks (now ranks the confounds
        too, matching the conventional "rank-then-partial" definition — a slight change
        from the old "residualize-then-rank"). Removed the misnamed numpy
        `semi_partial_correlation*` helpers. ~150 lines of branchy correlation code gone.
        Logistic GLM deliberately NOT used for selection (no closed form, FWL fails →
        would need batched IRLS); logistic stays in the model-fitting stage (`LinearCPM`).
  - [x] **Edge-stability significance — RESOLVED (see Open decisions #7).** Per-edge
        FDR (Benjamini–Yekutieli) and per-edge max-statistic both failed in practice:
        FDR needs raw p-values far below the `1/(n_perm+1)≈0.001` resolution floor, and
        the max-statistic is crippled by the discreteness of stability (only `K+1` values
        over `K` folds). Replaced both with a **subnetwork-level Network-Based Statistic**
        (`calculate_p_values_edges_nbs`, Zalesky et al. 2010): stability-threshold →
        connected components (`networkx`) → permutation max-component null; configurable
        `extent`/`intensity` component statistic. Added **network TFCE**
        (`calculate_p_values_edges_tfce`) as a threshold-free per-edge alternative.
        Selected via `CPMAnalysis(edge_significance_method=...)`; both write the same
        `[n_nodes, n_nodes, 2]` `stability_edges_significance.npy` so the report is
        unchanged. Old FDR/max-value methods removed. **Caveat documented:** NBS licenses
        subnetwork-level, not per-edge, claims.
  - [ ] **Continuous edge statistic for NBS/TFCE (deferred).** Stability is coarse;
        persist the per-edge semi-partial correlation (currently computed in
        `edge_selection.py` then discarded after thresholding), aggregate across folds into
        a continuous `[N_features, N_perms]` map, and let NBS/TFCE run on it instead of
        stability. Continuous → better-behaved thresholding/cluster-forming. Note the
        meaning shift ("consistently associated" vs "consistently selected") and memory
        cost (~140 MB at 35k edges × 1000 perms after folding across folds).
  - [x] **Residualization (`get_residuals`):** validated OLS residualization (intercept
        + pinv) against numpy `lstsq` for both data orientations ([N,features] and
        [batch,N]) to ~1e-9, incl. orthogonality of residuals to the confound space.
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
- [~] Run the full suite on macOS (arm64), Linux, Windows in CI. **Done via the matrix.**
      Two CI bugs fixed: (1) `example_simulated_data.py` had `cpm.run()` commented out
      while `generate_html_report()` ran → crashed on a clean checkout (passed locally
      only due to leftover `examples/tmp/` artifacts); fixed + switched to `device='cpu'`.
      (2) Windows runners hit `_tkinter.TclError` because matplotlib defaulted to the Tk
      GUI backend (broken Tcl/Tk on the runner); set `matplotlib.use("Agg")` in
      `tests/conftest.py`. macOS/Linux green; Windows re-running.
- [ ] **DEFERRED — matplotlib global backend (Windows robustness).** The library imports
      `matplotlib.pyplot` in several modules, so a real Windows user with broken/absent
      Tcl/Tk can hit `_tkinter.TclError` during `cpm.run()`/report generation. Workaround
      documented in `installation.md` (set `MPLBACKEND=Agg`). Proper fix (deferred,
      discussed w/ Nils): rewrite the report plotting to the object-oriented `Figure()`
      API instead of `plt.figure()`, so it renders to disk via Agg regardless of the
      user's backend WITHOUT a global `matplotlib.use("Agg")` (which would clobber a
      Jupyter user's inline backend on `import cccpm`).

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
- [x] Migrate the quickstart/example scripts to the new SEM-based simulator
      (`simulation/simulate_sem.py`: `simulate_data_given_R2` / `simulate_data_given_kappa`
      / `generate_confound_grid`) for more realistic, confound-aware example data with
      known ground-truth R². `regression_quickstart.py`, `classification_quickstart.py`
      (median-split of the continuous y), and `example_simulated_data.py` now use
      `simulate_data_given_kappa` (R2_X_y=0.4, kappa=0.3, mixed/pure-signal/confound-only
      edge classes). All three run; `test_integration.py`/`test_reporting.py` green.
      Also added a **Simulating Data** docs page (`documentation/docs/simulation.md`:
      generative model, edge classes, R²/kappa knobs, grid sweep, binarising the target)
      + mkdocstrings API page (`api/simulation.md`), both wired into nav; `mkdocs build
      --strict` passes.
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

**Done on branch `html-report-redesign` (off `develop`).** Replaced arakawa with a
custom Jinja2 + CSS report (self-contained, offline), then a full visual redesign:
- [x] Fast iteration loop: `scripts/preview_report.py` regenerates the report from a
      fixed fixture (`--results`, `--atlas`, `--no-open`) and opens it.
- [x] Self-contained & portable: all figures are inline vector **SVG** (brain renders
      base64 PNG); report works offline. New `plots/figure_style.py` standardises figure
      sizes + palette + SVG output (root-cause fix for mismatched/blurry figures). Report
      shrank 1.88 MB → ~0.8 MB.
- [x] **Hero + inverted-pyramid layout**: Summary (one-sentence verdict + stat chips +
      headline scatter) → Model Comparison (one faceted figure + APA table) → Predictions
      → Network Strengths → Brain & Edges → Stable Edges → Data & Methods appendix.
      Restored the dropped Hyperparameters section.
- [x] **Neuroscience figures written into CCCPM** (`plots/brain_figures.py`,
      `plots/connectome_utils.py`): connectivity matrix, network-summary matrix, chord
      (pycirclize), node-degree/hub — glass-brain kept. Designed for later extraction into
      `wwu-mmll/brainplots` (deferred). Added `tests/fixtures/atlas_30.csv` + smoke tests.
- [x] Explanatory captions embedded in each section (terminology aligned with
      `interpreting_results.md`).
- [~] Accessibility / print-friendliness: `@media print` stylesheet in place (hides
      sidebar, page-breaks sections); a full a11y audit is still deferred.
- [x] **Second design pass + Nils' review fixes (shipped in 0.3.1):** dropped the dark KPI
      block and the headline gradient (flat accent panel); three square connectome scatters
      (positive/negative/both) + covariates in the Summary, each annotated with r AND p
      (`set_box_aspect` makes them square); removed the now-redundant Predictions section;
      explanations are always-visible (`.info`, no dropdown); darker text tokens; full-width
      section intros; version+date pinned at the bottom of the left nav; config rendered as a
      table (not cards); removed the redundant appendix Data Summary; capped Stable-Edges
      tables (top 50/network, scrollable) — was ~30k px tall; design-token stylesheet; masthead.
      glass brain now renders from the stability matrices; hero r matches the scatter r.
      Docs updated (`interpreting_results.md` new section layout; embedded example report
      regenerated with atlas). **Note:** `wwu-mmll/brainplots` could not be attached via
      `add_repo` (workspace access-policy 404; repo is public/reachable over SSH).
- [x] **Edge-significance section overhaul (2026-07-01).** The Stable-Edges section now
      names the significance method (NBS/TFCE) and its parameters, reports per-network
      diagnostics (largest NBS component, # significant subnetworks/edges), and renders a
      permutation null-distribution plot per network (`plots/stats_figures.py`). Removed the
      top-50 cap — **all** significant edges are shown per network — and added a downloadable
      CSV (data-URI) of every selected edge with its stability + significance. Backed by a
      new `stability_edges_significance_meta.json` artifact (method, null distributions,
      components) written by the permutation step; loaded via
      `ReportDataLoader.load_edge_significance_meta`. Docs (`methods.md`,
      `interpreting_results.md`) updated; `mkdocs build --strict` passes.
- [ ] **Later (deferred, discussed w/ Nils):** move the brain figures into the
      `wwu-mmll/brainplots` toolbox and make it a PyPI package; then have CCCPM depend on
      it (likely an optional `cccpm[plots]` extra). Not done now — code lives in CCCPM.

## Phase 6 — Release
- [x] **0.3.0 RELEASED 2026-06-30** (install-reliability pass; see 0.3.0 CHANGELOG).
- [x] **0.3.1 RELEASED 2026-06-30** (HTML report redesign). Bumped pyproject 0.3.0→0.3.1 +
      CHANGELOG; merged `html-report-redesign`→`develop`→`main` (all three branches in sync at
      the same commit); pushed `main` (docs deploy) and tag `v0.3.1` (→ PyPI via publish.yml
      OIDC). Verified the wheel BEFORE tagging: `jinja2`+`pycirclize` declared, `arakawa`
      gone, and the Jinja `templates/*.j2` + `styles.css` + assets are included in the wheel
      (so `generate_html_report()` works on a clean install). 182 tests green; `mkdocs build
      --strict` passes. **CI not verified from the session (no `gh` CLI):** at push time the
      Actions runs (Publish to PyPI on `v0.3.1`, Build and Deploy Docs on `main`, Run Tests,
      Package Smoke) were all queued/in-progress — **Nils: confirm they went green, that
      0.3.1 is on PyPI, and the docs site redeployed.**
- [ ] Next time, consider a `v0.3.x-test` → TestPyPI dry-run before the real tag (skipped
      again here since 0.3.0 already validated the publish path).

---

## Open decisions (need Nils' input)
1. **torch/GPU packaging**: default torch + doc GPU, or CPU-default + `[gpu]` extra?
2. **Next version number**: 0.3.0 (proposed) given 0.2.1 is on PyPI?
3. **Python version ceiling**: support up to 3.13 now, or also 3.14?
4. **PEP 621 migration** for pyproject, or stay on Poetry's `[tool.poetry]` table?
5. **Reference dataset** for ground-truth validation — do we have one to validate against?
6. **Edge-selection p-value computation** — the production correlation/point-biserial
   p-values use a normal approximation to the t-distribution (kept for torch/CUDA GPU
   acceleration). It is anti-conservative at small N (~13% too low at N=20, negligible
   at N>=100). Options to weigh together:
   - (a) **Keep the normal approximation** — fully GPU/CUDA, but slightly anti-conservative
     at small N (lets in marginally more edges). Simplest.
   - (b) **More accurate on-GPU approximation** of the t-tail (e.g. a continued-fraction /
     regularized-incomplete-beta implementation in torch) — stays on GPU, much closer to
     exact. More code to write and validate.
     Note: `torch.special.betainc` is NOT available in torch 2.12, so this needs a
     hand-rolled implementation.
   - (c) **Exact via scipy `t.sf` on CPU** — exactly matches the standalone
     `*_with_pvalues` functions already tested to 1e-10, but adds a GPU->CPU roundtrip
     in edge selection. The p-values feed only a threshold comparison (not a hot loop),
     so the cost is small in practice.
   Recommendation to discuss: (b) if we want to stay GPU-pure with correct stats, else
   (c) scoped to just the p-value step (r is still computed on GPU).
   Note: confirmed `torch.special.betainc` is MISSING in torch 2.12.1 (but `igamma`,
   `ndtr`, `erf` exist). The unified OLS-GLM refactor does NOT change this — it uses the
   same normal-tail approximation; the question is unchanged and decoupled.
7. **Multiple-comparison / FDR correction in edge selection** (DISCUSSION PENDING).
   `PThreshold` supports statsmodels corrections (bonferroni/fdr_bh/…) applied to the
   flattened edge p-values, default `None`. Audited: no bug (the hard-coded `alpha=0.05`
   only affects the unused `reject` array; corrected p-values are compared to the user
   threshold). Open question: should CPM correct across the ~tens-of-thousands of edges
   by default, and with which method? CPM is traditionally run *uncorrected* at a liberal
   threshold (e.g. p<0.01) because the predictive model + permutation test provide the
   real inferential control — but this deserves a deliberate decision. Decide together.
   NB: this is the *edge-selection* threshold, distinct from *edge-stability
   significance* (resolved — NBS/TFCE, see Phase 1).
