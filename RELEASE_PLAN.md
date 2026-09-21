# CCCPM Release Readiness Plan

Goal: make `cccpm` easy and reliable for researchers to install and use for
connectome-based predictive modeling (CPM) across macOS / Linux / Windows and
Python 3.10–3.14, with trustworthy results, modern docs, and a polished HTML report.

Status legend: [ ] todo · [~] in progress · [x] done

**Last audited: 2026-07-22** against `0.4.2` (suite green). Completed work is
pruned from this plan — the record lives in git history and `CHANGELOG.md`.
Open items are grouped by category and ordered most-important-first, both between
and within sections.

---

## Correctness & statistical validity

- [ ] **Clear error when no confound is provided.** A missing/`None` `covariates`
      argument currently crashes deep in `check_data`/`get_residuals` instead of
      failing fast. Catch it early in `check_data`/`CPMAnalysis.run` and raise a
      clear message that a confound is currently required (the covariates/full/
      residuals variants and the `*_partial` statistics all assume one). Add a
      test. (Longer term: consider a real no-confound mode.)
- [~] **Edge-selection p-value approximation** (decision #6). The normal
      approximation to the t-tail in `correlations_and_pvalues` is
      anti-conservative at small N (~13% too low at N=20, negligible at N≥100).
      Pick (a) keep, (b) on-GPU exact t-tail, or (c) scipy `t.sf` on CPU for the
      threshold step only. Needs sign-off.
- [x] **Connected-component edge selection** (2026-07-22).
      `UnivariateEdgeSelection(connected_components=True|int)` drops selected edges
      not in a connected component with ≥ N edges (N=2 by default), per network,
      per fold and permutation (`filter_connected_components`, networkx). Tested.
- [ ] **External reference validation** vs Shen MATLAB / GenCPM on ≥1 dataset, so
      paper numbers are defensible (this is what the `cccpm_paper` benchmark is
      for — feed the result back here once numbers agree).
- [ ] **Classification path:** expand tests (probabilities, AUC, class imbalance,
      StratifiedKFold edge cases).
- [ ] **Verify MPS (Apple) / CUDA** run end-to-end on real hardware.

## Performance

- [x] **VRAM blow-up in edge-stability aggregation (GPU OOM)** (2026-07-22).
      Edge bookkeeping moved to CPU and reduced to a running fold-sum
      (`cv_edge_sum`); node×node densification built on CPU; per-fold `edges.npy`
      written for the real run only (permutations keep just the fold-averaged
      `stability_edges.npy`). Verified on a real GPU: 100 nodes × 10 folds × 200
      perms now peaks at ~69 MB VRAM for this step. Small-GPU guidance added to
      `installation.md`.
- [x] **GPU-vs-CPU speed test + investigation** (2026-07-22).
      `scripts/benchmark_cpu_gpu.py` times edge selection + model fit on CPU vs
      CUDA across sizes/permutations. **Finding:** CPU≈GPU is expected at
      `perms=1` (a plain run) — the matmuls are tiny and GPU launch/transfer
      overhead cancels the win; the GPU only pulls ahead with a large permutation
      batch and/or big connectomes (e.g. ~23× at 200 nodes × 1000 perms on an
      RTX 3090, but ~0.95× at 100 nodes × 1 perm). A full `run()` also has fixed
      CPU-bound overhead (IO, report, networkx) that dilutes the kernel speedup.

## Code health & cleanup
*Done 2026-07-22 (no behavior change, suite green): removed the dead/broken
`utils.py` converters + duplicate import; `ResultsManager.collect_results` and
`_save_inner_cv_to_csv`; the `SelectPercentile`/`SelectKBest` stubs; the duplicated
`PThreshold` docstring; the unused `simulation/simulate_multivariate` module; the
AI-narration comments in `linear_model.py`; and the double `cv_predictions.csv`
write. The 3 remaining test-only matrix/vector converters were kept — they are the
reference implementations that validate the production `vector_to_matrix_tensor_version`.*

No open code-health items. (The duplicate `vector_to_upper_triangular_matrix` in
`plots/cpm_chord_plot.py` was unused and has been deleted; the surviving converters
live in `connectome.py`.)

## Packaging & cross-platform install

- [ ] **torch install strategy** (decision #1): keep default torch + document GPU,
      or CPU-default + a `cccpm[gpu]` extra.
- [ ] Set version floors for numpy (1.x vs 2.x), pandas, scikit-learn, nilearn;
      verify under numpy 2.x.
- [ ] Decide whether to commit `poetry.lock` (CI reproducibility) and whether to
      migrate pyproject to PEP 621 (decision #4).
- [ ] Sanity-check heavy report deps (`netplotbrain`, `scikit-image`,
      `pycirclize`) on Windows; consider an optional `cccpm[plots]` extra.
- [ ] **matplotlib backend (Windows):** rewrite report plotting to the OO
      `Figure()` API so it renders via Agg regardless of the user's backend,
      without a global `matplotlib.use("Agg")` (workaround documented in
      `installation.md`: `MPLBACKEND=Agg`).
- [ ] **Release:** do a `vX.Y.Z-test` → TestPyPI dry-run, clean-install on all 3
      OSes, then tag for PyPI. (Nils triggers deployment.)

## Docs & report

- [x] **Brain-plot edge thresholding + in-report selector** (2026-07-22). Brain &
      Edges defaults to significant edges (NBS/TFCE p<alpha) with an in-report
      button group to switch the matrix/hub/chord views between Significant / Top
      5% / Top 10% (`masked_signed_stability_matrix` + self-contained JS/CSS).
      Glass brain renders the default subset. *Follow-up: make the glass brain
      switch too if netplotbrain render cost is addressed.*
- [ ] Show key variations in both quickstarts: confound control (partial vs
      residuals), nested CV with p-threshold tuning, stable-edge selection,
      permutation testing, and passing `atlas` for brain plots.
- [ ] Add in-report captions; finish the accessibility/print audit.
- [ ] Optional: a real-data (or realistic simulated) end-to-end tutorial.
- [ ] Optional: extract the brain figures into `wwu-mmll/brainplots`, publish it,
      and depend on it via `cccpm[plots]` (code currently lives in CCCPM).
- [~] `examples/` curation: `example_simulated_classification.py` overlaps the
      quickstarts — which author-written examples to keep is a call for Nils.

---

## Open decisions (need Nils' input)
1. **torch/GPU packaging**: default torch + doc GPU, or CPU-default + `[gpu]` extra?
4. **PEP 621 migration** for pyproject, or stay on Poetry's `[tool.poetry]` table?
6. **Edge-selection p-value computation**: keep the GPU normal approximation (a),
   a hand-rolled on-GPU exact t-tail (b; `torch.special.betainc` is missing in
   torch 2.x), or scipy `t.sf` on CPU for the p-value step only (c). Recommend
   (b) to stay GPU-pure, else (c).
7. **Multiple-comparison / FDR in edge selection**: `PThreshold` supports
   statsmodels corrections (default `None`, no bug). Should CPM correct across the
   ~tens-of-thousands of edges by default, and with which method? CPM is
   traditionally run uncorrected at a liberal threshold because the model +
   permutation test provide the real inferential control — decide deliberately.
   (Distinct from edge-*stability* significance, resolved via NBS/TFCE.)

*Released: 0.3.0 (install reliability), 0.3.1 (HTML report redesign), 0.4.0
(NBS/TFCE edge-stability significance), 0.4.1 (increment baseline + TFCE fixes),
0.4.2 (built-in atlas registry).*
