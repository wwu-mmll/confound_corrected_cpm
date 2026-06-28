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

- [ ] Review coverage report (`pytest --cov --cov-report=html`), find untested paths.
- [ ] Audit `test_ground_truth.py`: confirm assertions check *actual numbers/edges*,
      not just "runs without error". Tighten thresholds where weak.
- [ ] Add a frozen-baseline regression test: run a fixed seed end-to-end, snapshot
      key outputs (metrics, selected edges, permutation p-values), fail on drift.
- [ ] Validate against an external reference (Shen et al. CPM / existing MATLAB
      results) on at least one dataset so numbers are defensible in a paper.
- [ ] Numerical correctness spot-checks: partial correlation, FDR correction,
      permutation p-value definition (off-by-one / +1 convention), residualization.
- [x] **Input validation gap (found while writing examples):** `check_data()` now
      validates that `n_features` is a valid upper-triangular connectome size
      (`infer_n_nodes`) and raises a clear error suggesting the nearest valid sizes,
      instead of crashing deep in edge stability. Covered by new tests.
- [ ] Classification path: expand beyond current tests (probabilities, AUC, class
      imbalance, StratifiedKFold edge cases).
- [ ] Determinism: `CPMAnalysis.__init__` calls global `np.random.seed(42)` /
      `torch.manual_seed(42)` — this mutates the user's global RNG as a side effect.
      Decide on a local RNG / `random_state` param instead.
- [ ] Device default mismatch: `LinearCPM.__init__` defaults `device='cuda'` while
      `CPMAnalysis` defaults `'cpu'`. Make consistent; verify CPU/MPS/CUDA all work.
- [ ] Run the full suite on macOS (arm64), Linux, Windows locally or in CI.

## Phase 2 — Code structure / modularization
Mostly in good shape (reporting + models are already split). Targeted cleanups:

- [x] Improve top-level public API in `src/cccpm/__init__.py`: now exports
      `UnivariateEdgeSelection`, `PThreshold`, `TaskType`, models, and `__version__`.
- [x] Update stale `CLAUDE.md` key-modules table (`pytorch_model.py` ->
      `models/linear_model.py`, added nonlinear models row).
- [ ] Remove the global RNG side effect (ties to Phase 1 determinism).
- [ ] Clean `examples/`: dedupe `mediator_sim_example.py` vs
      `mediator_simulation_example.py`, drop `tmp/`, `.ipynb_checkpoints/`, `.DS_Store`
      (add to .gitignore). Keep a small, curated set of runnable examples.
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
- [ ] Add a clean-environment install smoke test to CI (build wheel, install in a
      fresh venv, import, run a tiny example) — catches the missing-torch class of bug.

## Phase 4 — Documentation overhaul
Outdated and incomplete. Make it the on-ramp for researchers.

- [x] Fix README: corrected install, quick-start (`from cccpm import CPMAnalysis`,
      string `edge_statistic`), and repo badges.
- [x] Fix `getting_started.md`: `CPMRegression` -> `CPMAnalysis`, imports, the
      `edge_statistic` list/string bug, and the `estimate` -> `run` method name.
- [ ] Rewrite `installation.md`: per-OS instructions, Python version notes, torch/GPU
      guidance, troubleshooting (the arm64/Rosetta + torch wheel issue we hit).
- [ ] Rewrite `getting_started.md`: replace `CPMRegression` → `CPMAnalysis`, fix
      imports and the `edge_statistic` list/string bug, verify every snippet runs.
- [ ] New: conceptual docs explaining the method — edge selection, confound control
      (partial correlation vs residualization), nested CV, stable edges, permutation
      testing, the four model variants (connectome/covariates/full/residuals),
      network strengths, and how to interpret outputs.

### Paired regression + classification examples (first-class deliverable)
Every concept gets a runnable script in `examples/` **and** a matching docs tutorial,
in both flavors so users can copy the one matching their task.
- [x] `examples/regression_quickstart.py` — minimal, well-commented, simulated data.
- [x] `examples/classification_quickstart.py` — minimal, well-commented, simulated data.
- [x] Both wired into `test_integration.py` so CI verifies they run end-to-end.
- [ ] Mirror both as docs tutorials (regression + classification), every snippet
      verified to run (covered by `test_integration.py`).
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
- [ ] Add `CHANGELOG.md`, `CONTRIBUTING.md`, `CITATION.cff` (researchers will cite it).

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
