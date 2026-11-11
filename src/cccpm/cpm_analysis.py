"""`CPMAnalysis`: the main entry point orchestrating the CPM pipeline."""
import os
import json
import logging
import warnings

from typing import Union, Type

import torch
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit, KFold, RepeatedKFold, StratifiedKFold

from cccpm.inner_fold import run_inner_folds
from cccpm.logging import setup_logging
from cccpm.models.linear_model import LinearCPM
from cccpm.edge_selection import UnivariateEdgeSelection, PThreshold
from cccpm.results_manager import ResultsManager
from cccpm.inference import PermutationManager
from cccpm.preprocessing import (to_device, torch_train_test_split, torch_impute_missing_values,
                                 residualize_train_test, select_stable_edges)
from cccpm.validation import (check_data, detect_task_type, validate_task_type,
                              infer_n_nodes)
from cccpm.memory import plan_permutation_chunk
from cccpm.atlases import resolve_atlas
from cccpm.scoring import score_models
from cccpm.reporting import HTMLReporter
from cccpm.reporting.data_insights import generate_data_insights
from cccpm.constants import Models, Networks, TaskType


# Renamed in 0.7.0: "edge significance"/"nbs_threshold" were ambiguous with the
# PThreshold p-value and with "a fraction of folds, not a p-value" respectively.
_RENAMED_IN_0_7_0 = {
    'edge_significance_method': (
        'stability_significance_method',
        "it sets how edge *stability* significance is established from the "
        "permutations, not how edges are selected"),
    'nbs_threshold': (
        'nbs_stability_threshold',
        "it is a stability threshold -- a fraction of folds -- not a p-value"),
}


class CPMAnalysis:
    """Orchestrates CPM analysis with cross-validation and permutation testing."""
    def __init__(self,
                 results_directory: str,
                 task_type: Union[TaskType, str, None] = None,
                 cpm_model: Type[LinearCPM] = LinearCPM,
                 cv: Union[BaseCrossValidator, BaseShuffleSplit, RepeatedKFold, StratifiedKFold] = KFold(n_splits=10, shuffle=True, random_state=42),
                 inner_cv: Union[BaseCrossValidator, BaseShuffleSplit, RepeatedKFold, StratifiedKFold] = None,
                 edge_selection: UnivariateEdgeSelection = UnivariateEdgeSelection(
                     selection_statistic='pearson',
                     edge_selection=[PThreshold(threshold=[0.05], correction=[None])]
                 ),
                 select_stable_edges: bool = False,
                 stability_threshold: float = 0.8,
                 impute_missing_values: bool = True,
                 model_input: str = 'raw',
                 n_permutations: int = 0,
                 stability_significance_method: str = "nbs",
                 nbs_stability_threshold: float = 0.5,
                 nbs_component_stat: str = "extent",
                 atlas: str = None,
                 atlas_labels: str = None,
                 device: str = 'cpu',
                 random_state: int = 42,
                 **removed):
        """Configures the CPM pipeline; call ``run()`` to execute it.

        Args:
            task_type: ``'regression'`` or ``'classification'``; auto-detected
                from ``y`` if ``None``.
            cpm_model: Backend class, e.g. ``LinearCPM``/``DecisionTreeCPM``/
                ``RandomForestCPM``/``GAMCPM``.
            inner_cv: If ``None``, no inner loop runs and exactly one
                edge-selection configuration must be provided.
            select_stable_edges: Keep only edges selected in a sufficient
                fraction of inner folds (``stability_threshold``); requires
                ``inner_cv``.
            model_input: ``'raw'`` feeds the connectome as-is to the models;
                ``'residualized'`` regresses covariates out of it first
                (fit on train, applied to test), independently of
                ``selection_input`` on the edge statistic. This is a run-level
                choice rather than an extra model because OLS backends are
                invariant to it (`full`/`increment` unchanged) but non-linear
                backends are not -- on identical edges `full` moves by 391% of
                sd(y) for ``DecisionTreeCPM``, 65% for ``RandomForestCPM`` and
                19% for ``GAMCPM``.
            n_permutations: ``0`` disables permutation testing; use 1000+ for
                publishable p-values.
            stability_significance_method: ``'nbs'`` (connected-component
                test) or ``'tfce'`` (per-edge FWER control) -- how edge
                *stability* across folds is tested against the permutation
                null, distinct from the ``PThreshold`` selection cutoff.
            nbs_stability_threshold: Fraction-of-folds cutoff (not a p-value)
                for NBS component forming; ignored for ``'tfce'``.
            nbs_component_stat: ``'extent'`` (edge count) or ``'intensity'``
                (summed stability) NBS component statistic; ignored for
                ``'tfce'``.
            atlas: Built-in atlas name (see :func:`cccpm.atlases.list_atlases`)
                or a custom CSV with ``region``, ``x``, ``y``, ``z`` and
                optionally ``network``/``hemisphere``/``structure`` columns;
                a ``network`` column auto-enables the network-summary plots.
            atlas_labels: Deprecated alias for ``atlas`` (custom CSV path
                only); ``atlas`` wins if both are given.
            device: ``'cpu'``, or ``'cuda'``/``'gpu'`` (falls back to CPU with
                a warning if unavailable).
            random_state: Seeds a local RNG for permutations; does not touch
                the global NumPy/torch RNG.
        """
        for old, (new, why) in _RENAMED_IN_0_7_0.items():
            if old in removed:
                raise TypeError(
                    f"{old!r} was renamed to {new!r} in 0.7.0, because {why}. "
                    f"Pass {new}={removed[old]!r} instead.")
        if removed:
            raise TypeError(
                f"{type(self).__name__}() got an unexpected keyword argument "
                f"{sorted(removed)[0]!r}.")

        self.results_directory = results_directory

        if isinstance(task_type, str):
            task_type = TaskType(task_type)
        self.task_type = task_type
        self.random_state = random_state
        os.makedirs(self.results_directory, exist_ok=True)
        os.makedirs(os.path.join(self.results_directory, "edges"), exist_ok=True)
        os.makedirs(os.path.join(self.results_directory, "permutation"), exist_ok=True)
        setup_logging(os.path.join(self.results_directory, "cpm_log.txt"))
        self.logger = logging.getLogger(__name__)

        self.cpm_model = cpm_model
        self.cv = cv
        self.inner_cv = inner_cv
        self.edge_selection = edge_selection
        self.select_stable_edges = select_stable_edges
        self.stability_threshold = stability_threshold
        self.impute_missing_values = impute_missing_values
        if model_input not in ('raw', 'residualized'):
            raise ValueError(
                f"model_input must be 'raw' or 'residualized', got {model_input!r}.")
        self.model_input = model_input


        self.n_permutations = n_permutations
        self.stability_significance_method = stability_significance_method
        self.nbs_stability_threshold = nbs_stability_threshold
        self.nbs_component_stat = nbs_component_stat

        if device.lower() == 'gpu' or device.lower() == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.logger.warning("CUDA or GPU not available, using CPU instead.")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

        self._log_analysis_details()

        if self.inner_cv is None:
            if len(self.edge_selection.param_grid) > 1:
                raise RuntimeError("Multiple hyperparameter configurations but no inner cv defined. "
                                   "Please provide only one hyperparameter configuration or an inner cv.")
            if self.select_stable_edges:
                raise RuntimeError("Stable edges can only be selected when using an inner cv.")

        self.atlas = self._resolve_atlas(atlas, atlas_labels)
        self.atlas_labels = self._save_atlas(self.atlas)

        self.results_manager = None

    def _log_analysis_details(self):
        task_name = "CPM Classification" if self.task_type == TaskType.classification else "CPM Regression"
        self.logger.info(f"Starting {task_name} Analysis")
        self.logger.info("="*50)
        self.logger.info(f"Results Directory:       {self.results_directory}")
        self.logger.info(f"Task Type:               {self.task_type.value if self.task_type else 'Auto-detect'}")
        self.logger.info(f"CPM Model:               {self.cpm_model.name}")
        self.logger.info(f"Outer CV strategy:       {self.cv}")
        self.logger.info(f"Inner CV strategy:       {self.inner_cv}")
        self.logger.info(f"Edge selection method:   {self.edge_selection}")
        self.logger.info(f"Select stable edges:     {'Yes' if self.select_stable_edges else 'No'}")
        if self.select_stable_edges:
            self.logger.info(f"Stability threshold:     {self.stability_threshold}")
        self.logger.info(f"Impute Missing Values:   {'Yes' if self.impute_missing_values else 'No'}")
        self.logger.info(f"Selection input:         {self.edge_selection.statistic._input}")
        self.logger.info(f"Model input:             {self.model_input}")
        self.logger.info(f"Number of Permutations:  {self.n_permutations}")
        self.logger.info(f"Device:                  {self.device}")
        self.logger.info("="*50)

    def _resolve_atlas(self, atlas, atlas_labels):
        """Resolves ``atlas``/deprecated ``atlas_labels`` to a validated DataFrame (or ``None``)."""
        if atlas_labels is not None:
            warnings.warn(
                "`atlas_labels` is deprecated and will be removed in a future "
                "release; use `atlas` instead (it accepts both built-in atlas "
                "names and custom CSV paths).",
                DeprecationWarning,
                stacklevel=3,
            )
            if atlas is None:
                atlas = atlas_labels
            else:
                self.logger.warning(
                    "Both `atlas` and `atlas_labels` were provided; using "
                    "`atlas` and ignoring `atlas_labels`."
                )

        atlas_df = resolve_atlas(atlas)
        if atlas_df is not None:
            self.logger.info(f"Using atlas '{atlas}' with {len(atlas_df)} regions.")
        return atlas_df

    def _save_atlas(self, atlas_df):
        """Persists the resolved atlas to ``<results_directory>/edges/atlas.csv``; returns the path or ``None``."""
        if atlas_df is None:
            return None
        dest_path = os.path.join(self.results_directory, "edges", "atlas.csv")
        atlas_df.to_csv(dest_path, index=False)
        self.logger.info(f"Saved atlas to {dest_path}")
        return dest_path

    def _validate_atlas_node_count(self, n_features):
        """Raises if the atlas region count doesn't match the implied node count (would silently mislabel plots)."""
        if self.atlas is None:
            return
        n_nodes = infer_n_nodes(n_features)
        if n_nodes is not None and len(self.atlas) != n_nodes:
            raise ValueError(
                f"Atlas has {len(self.atlas)} regions but the connectome has "
                f"{n_features} edges, implying {n_nodes} nodes. The atlas must "
                f"have exactly one row per connectome node. Please select an "
                f"atlas matching your parcellation."
            )

    def _available_models(self):
        """Without covariates only ``connectome`` is meaningful (``full``/``increment`` would be degenerate)."""
        if getattr(self, 'has_covariates', True):
            return [m.name for m in Models]
        return [Models.connectome.name]

    def _validate_covariate_requirements(self, covariates):
        """Raises if an option needing covariates was passed with none.

        Without this the run would quietly degrade instead of crashing: a
        ``*_partial`` statistic with an empty confound design is just the
        plain statistic, silently answering a different question.
        """
        if covariates is not None:
            return

        offenders = []
        if self.edge_selection.statistic._input == 'residualized':
            offenders.append(
                "selection_input='residualized' (confound-controlled edge "
                "selection needs confounds; use selection_input='raw')")
        if self.model_input == 'residualized':
            offenders.append(
                "model_input='residualized' (there is nothing to residualise "
                "the connectome against; use model_input='raw')")

        if offenders:
            raise ValueError(
                "covariates=None, but these options require covariates: "
                + "; ".join(offenders) + "."
            )

    def run(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, pd.DataFrame, np.ndarray],
            covariates: Union[pd.Series, pd.DataFrame, np.ndarray, None] = None):
        """Fits the model, then (if ``n_permutations > 0``) runs permutation testing.

        Args:
            covariates: Omit (``None``) for vanilla CPM with no confound
                control -- only ``connectome`` is defined, the rest are NaN
                (see ``available_models.json``), and options presupposing
                covariates (a ``*_partial`` statistic, ``model_input=
                'residualized'``) raise up front.
        """
        self.logger.info("Starting CPM estimation.")

        self._validate_covariate_requirements(covariates)

        generate_data_insights(X=X, y=y, covariates=covariates, results_directory=self.results_directory)
        X, y, covariates = check_data(X, y, covariates, impute_missings=self.impute_missing_values)

        self.has_covariates = covariates is not None
        if not self.has_covariates:
            self.logger.info(
                "No covariates supplied: running vanilla CPM. Only the "
                "'connectome' model is defined; covariates/full/"
                "increment are reported as NaN.")
            # Zero-width design keeps downstream tensor ops valid without a
            # None check in each of them.
            covariates = np.empty((X.shape[0], 0), dtype=np.float64)

        self._validate_atlas_node_count(X.shape[1])

        if self.task_type is None:
            self.task_type = detect_task_type(y)
            self.logger.info(f"Auto-detected task type: {self.task_type.value}")
        else:
            validate_task_type(y, self.task_type)
            self.logger.info(f"Using specified task type: {self.task_type.value}")

        with open(os.path.join(self.results_directory, 'task_type.txt'), 'w', encoding='utf-8') as f:
            f.write(self.task_type.value)

        # Written to disk (not just logged) so the report can show which
        # confound-control cell produced it even if logging is turned down.
        with open(os.path.join(self.results_directory, 'run_config.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'selection_statistic': self.edge_selection.statistic._statistic,
                'selection_input': self.edge_selection.statistic._input,
                'model_input': self.model_input,
                'has_covariates': self.has_covariates,
            }, f)

        self._single_run(X=X, y=y.reshape(-1, 1), covariates=covariates, perm_run=False)
        self.logger.info("=" * 50)

        if self.n_permutations > 0:
            self.logger.info(f"Running {self.n_permutations} permutations.")
            y_perms = self._create_permuted_y(y)
            self._single_run(X=X, y=y_perms, covariates=covariates, perm_run=True)
            PermutationManager.calculate_permutation_results(
                self.results_directory, self.logger,
                method=self.stability_significance_method,
                nbs_stability_threshold=self.nbs_stability_threshold,
                nbs_component_stat=self.nbs_component_stat)

        self.logger.info("=" * 50)
        self.logger.info("Estimation completed.")
        self.logger.info("Generating results file.")
        reporter = HTMLReporter(results_directory=self.results_directory, atlas_labels=self.atlas_labels)
        reporter.generate_html_report()

    def generate_html_report(self):
        """(re)generates the HTML report from the results already on disk."""
        self.logger.info("Generating HTML report.")
        reporter = HTMLReporter(results_directory=self.results_directory, atlas_labels=self.atlas_labels)
        reporter.generate_html_report()

    def _create_permuted_y(self, y):
        """Builds ``n_permutations`` row-wise permutations of `y` as a [N_samples, N_perms] array."""
        y_tensor = to_device(y, "cpu")
        y_matrix = y_tensor.unsqueeze(0).expand(self.n_permutations, -1)

        generator = torch.Generator(device=y_matrix.device).manual_seed(self.random_state)
        noise = torch.rand(y_matrix.shape, generator=generator,
                           dtype=y_matrix.dtype, device=y_matrix.device)
        indices = noise.argsort(dim=1)

        permuted = y_matrix.gather(1, indices)
        return permuted.t().numpy()

    def _single_run(self, X, y, covariates, perm_run: bool = False):
        """Runs cross-validation over all outer folds (real data or permuted targets) and aggregates results."""
        if perm_run:
            results_directory = os.path.join(self.results_directory, "permutation")
        else:
            results_directory = self.results_directory

        # store_fold_edges=False on permutation runs: keeping per-fold masks
        # there would be a [Features, 2, Folds, n_permutations] tensor.
        results_manager = ResultsManager(output_dir=results_directory, n_runs=y.shape[1],
                                         n_folds=self.cv.get_n_splits(), n_features=X.shape[1],
                                         device=self.device, store_fold_edges=not perm_run,
                                         available_models=self._available_models())

        # RepeatedKFold enumerates all folds of repeat 0, then repeat 1, etc.
        n_repeats = getattr(self.cv, 'n_repeats', 1)
        splits_per_repeat = max(self.cv.get_n_splits() // n_repeats, 1)

        # self.cv.split still runs on the original CPU X/y -- some splitters
        # (e.g. StratifiedKFold) aren't guaranteed to accept a CUDA tensor.
        X_dev = to_device(X, self.device)
        y_dev = to_device(y, self.device)
        cov_dev = to_device(covariates, self.device)

        # Caps in-flight permutation columns so memory degrades to slower
        # chunking instead of OOM; real runs have n_runs == 1 and never engage.
        n_runs = y.shape[1]
        chunk = plan_permutation_chunk(n_features=X.shape[1], n_samples=X.shape[0],
                                       n_runs=n_runs, device=self.device)
        if chunk < n_runs:
            self.logger.info(
                f"Processing {n_runs} permutations in {-(-n_runs // chunk)} chunks of "
                f"at most {chunk} to stay within available memory.")

        splits = list(self.cv.split(X, y[:, 0]))
        iterator = tqdm(total=self.cv.get_n_splits() * (-(-n_runs // chunk)),
                        desc="Running outer folds", unit="fold")
        for run_start in range(0, n_runs, chunk):
            run_idx = slice(run_start, min(run_start + chunk, n_runs))
            y_chunk = y_dev[:, run_idx]
            for outer_fold, (train, test) in enumerate(splits):
                repeat = outer_fold // splits_per_repeat
                self._run_outer_fold(outer_fold, repeat, train, test, X_dev, y_chunk,
                                     cov_dev, results_manager, perm_run, run_idx=run_idx)
                iterator.update(1)
        iterator.close()

        results_manager.calculate_final_cv_results(task_type=self.task_type)
        results_manager.calculate_edge_stability()

        if not perm_run:
            self.logger.info(results_manager.agg_results.round(4).to_string())
            results_manager.save_predictions()
            results_manager.save_network_strengths()
            self.results_manager = results_manager

    def _run_outer_fold(self, outer_fold, repeat, train, test, X, y, covariates,
                        results_manager, perm_run, run_idx=slice(None)):
        """Runs edge selection, model fit, and scoring for one outer fold, storing the results."""
        X_train, X_test, y_train, y_test, cov_train, cov_test = torch_train_test_split(
            train, test, X, y, covariates)
        if self.impute_missing_values:
            X_train, X_test, cov_train, cov_test = torch_impute_missing_values(
                X_train, X_test, cov_train, cov_test)

        # Selection always sees the connectome as supplied; selection_input
        # (inside the statistic) decides whether it controls for covariates.
        edges = self._select_edges(X_train, y_train, cov_train, results_manager, outer_fold)
        results_manager.store_edges(param_idx=0, fold_idx=outer_fold, edges_tensor=edges,
                                    run_idx=run_idx)

        # Deliberately after selection, so selection_input and model_input stay independent.
        if self.model_input == 'residualized':
            X_train, X_test = residualize_train_test(
                X_train, X_test, cov_train, cov_test)

        model = self.cpm_model(edges=edges, device=self.device, task_type=self.task_type)
        model.fit(X_train, y_train, cov_train)

        y_pred = model.predict(X_test, cov_test, return_proba=True)

        if not perm_run:
            results_manager.store_predictions(y_pred=y_pred, y_true=y_test,
                                              fold=outer_fold, test_indices=test,
                                              repeat=repeat)
            network_strengths = model.get_network_strengths(X_test, cov_test)
            results_manager.store_network_strengths(network_strengths=network_strengths,
                                                    y_true=y_test, fold=outer_fold,
                                                    test_indices=test, repeat=repeat)

        metrics = score_models(y_true=y_test, y_pred=y_pred,
                               task_type=self.task_type, device=self.device)
        results_manager.store_metrics(param_idx=0, fold_idx=outer_fold,
                                      metrics_tensor=metrics, run_idx=run_idx)

    def _select_edges(self, X_train, y_train, cov_train, results_manager, outer_fold):
        """Returns the [N_features, 2, N_runs] edge mask for this fold, via inner-CV search or the fixed threshold."""
        if self.inner_cv:
            best_params, stability_edges = run_inner_folds(
                cpm_model=self.cpm_model,
                X=X_train,
                y=y_train,
                covariates=cov_train,
                inner_cv=self.inner_cv,
                edge_selection=self.edge_selection,
                results_directory=os.path.join(
                    results_manager.results_directory, 'folds', str(outer_fold)),
                device=self.device,
                task_type=self.task_type,
            )
            if self.select_stable_edges:
                return select_stable_edges(stability_edges, self.stability_threshold)
        else:
            best_params = [self.edge_selection.param_grid[0]] * y_train.shape[1]

        r_edges, t_edges, df = self.edge_selection.statistic.fit_transform(
            X=X_train, y=y_train, covariates=cov_train, device=self.device)
        self.edge_selection.r_edges = r_edges
        self.edge_selection.t_edges = t_edges
        self.edge_selection.df = df

        # return_selected_edges gives [Features, 2, N_thresholds, Runs]; with a
        # single configuration set, N_thresholds == 1.
        if all(params == best_params[0] for params in best_params):
            self.edge_selection.set_params(**best_params[0])
            return self.edge_selection.return_selected_edges()[:, :, 0, :]

        # The inner CV picked a different configuration for different runs, so
        # each run is thresholded with its own winner and reassembled.
        edges = torch.zeros(X_train.shape[1], len(Networks) - 1, len(best_params),
                            dtype=torch.bool, device=self.device)
        for run_id, params in enumerate(best_params):
            self.edge_selection.set_params(**params)
            self.edge_selection.r_edges = r_edges[:, [run_id]]
            self.edge_selection.t_edges = t_edges[:, [run_id]]
            edges[:, :, run_id] = self.edge_selection.return_selected_edges()[:, :, 0, 0]
        return edges
