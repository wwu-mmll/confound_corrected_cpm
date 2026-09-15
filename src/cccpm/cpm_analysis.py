import os
import logging
import warnings

from typing import Union, Type

import torch
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit, KFold, RepeatedKFold, StratifiedKFold
from sklearn.linear_model import LinearRegression

from cccpm.inner_fold import run_inner_folds
from cccpm.logging import setup_logging
from cccpm.models.linear_model import LinearCPM
from cccpm.edge_selection import UnivariateEdgeSelection, PThreshold, resolve_presence_threshold
from cccpm.results_manager import ResultsManager, PermutationManager
from cccpm.utils import (train_test_split, torch_train_test_split, check_data, impute_missing_values, torch_impute_missing_values,
                         torch_impute_missing_values_batched, build_fold_batch, residualize_train_test,
                         select_stable_edges, generate_data_insights, detect_task_type,
                         validate_task_type, infer_n_nodes)
from cccpm.atlases import resolve_atlas
from cccpm.scoring import score_models, score_models_batched
from cccpm.reporting import HTMLReporter
from cccpm.constants import Networks, TaskType
from cccpm.batch_planning import plan_batch_sizes, make_cpm_cost_fn, available_memory_bytes


class CPMAnalysis:
    """
    This class handles the process of performing CPM analysis with cross-validation and permutation testing.

    Supports both regression and binary classification tasks.
    """
    def __init__(self,
                 results_directory: str,
                 task_type: Union[TaskType, str, None] = None,
                 cpm_model: Type[LinearCPM] = LinearCPM,
                 cv: Union[BaseCrossValidator, BaseShuffleSplit, RepeatedKFold, StratifiedKFold] = KFold(n_splits=10, shuffle=True, random_state=42),
                 inner_cv: Union[BaseCrossValidator, BaseShuffleSplit, RepeatedKFold, StratifiedKFold] = None,
                 edge_selection: UnivariateEdgeSelection = UnivariateEdgeSelection(
                     edge_statistic='pearson',
                     edge_selection=[PThreshold(threshold=[0.05], correction=[None])]
                 ),
                 select_stable_edges: bool = False,
                 stability_threshold: float = 0.8,
                 impute_missing_values: bool = True,
                 calculate_residuals: bool = False,
                 n_permutations: int = 0,
                 edge_significance_method: str = "nbs",
                 nbs_threshold: float = 0.5,
                 nbs_component_stat: str = "extent",
                 atlas: str = None,
                 atlas_labels: str = None,
                 device: str = 'cpu',
                 random_state: int = 42):
        """
        Initialize the CPMAnalysis object.

        Parameters
        ----------
        results_directory: str
            Directory to which all results, plots, and the HTML report are saved.
        task_type: TaskType, str, or None, default=None
            Type of task: ``'regression'`` or ``'classification'``. If ``None``,
            it is auto-detected from the target variable (a binary target is
            treated as classification).
        cpm_model: type, default=LinearCPM
            The CPM model class to fit each fold. One of ``LinearCPM``,
            ``DecisionTreeCPM``, ``RandomForestCPM``, or ``GAMCPM``.
        cv: BaseCrossValidator or BaseShuffleSplit, default=KFold(10, shuffle=True)
            Outer cross-validation strategy used for performance estimation.
        inner_cv: BaseCrossValidator or BaseShuffleSplit, default=None
            Inner cross-validation strategy for hyperparameter tuning (e.g. the
            p-threshold) and stable-edge selection. If ``None``, no inner loop is
            run and exactly one edge-selection configuration must be provided.
        edge_selection: UnivariateEdgeSelection
            Edge-selection method and its hyperparameter grid.
        select_stable_edges: bool, default=False
            If ``True``, keep only edges selected in a sufficient fraction of inner
            folds (see ``stability_threshold``). Requires an ``inner_cv``.
        stability_threshold: float, default=0.8
            Minimum fraction of inner folds in which an edge must be selected to be
            considered stable. Only used when ``select_stable_edges=True``.
        impute_missing_values: bool, default=True
            Whether to impute missing values in ``X`` and the covariates (NaNs in
            the target ``y`` always raise an error).
        calculate_residuals: bool, default=False
            If ``True``, regress the covariates out of the connectome before
            modeling (residualization), in addition to the model variants.
        n_permutations: int, default=0
            Number of label permutations for significance testing. ``0`` disables
            permutation testing; use 1000+ for publishable p-values.
        edge_significance_method: str, default='nbs'
            How edge-stability significance is established from the permutations.
            ``'nbs'`` uses the Network-Based Statistic (connected-component test,
            subnetwork-level FWER control); ``'tfce'`` uses network Threshold-Free
            Cluster Enhancement (per-edge FWER control, no primary threshold).
        nbs_threshold: float, default=0.5
            Stability threshold (``>=``) for NBS component forming. Because
            stability is discrete over the outer folds, ``0.5`` keeps edges
            selected in a majority of folds. Ignored when method is ``'tfce'``.
        nbs_component_stat: str, default='extent'
            NBS component statistic: ``'extent'`` (number of edges, classic NBS)
            or ``'intensity'`` (summed supra-threshold stability). Ignored when
            method is ``'tfce'``.
        atlas: str, default=None
            Which atlas to use for the brain plots in the report. Either the name
            of a built-in atlas (e.g. ``'Schaefer100-17'``; see
            :func:`cccpm.atlases.list_atlases`) or a path to a custom CSV file
            with columns ``region``, ``x``, ``y``, ``z`` (MNI coordinates), and
            optionally ``network``, ``hemisphere``, ``structure``. When a
            ``network`` column is present, the network-summary matrix and chord
            diagram are enabled automatically. ``None`` disables the brain plots
            that require coordinates.
        atlas_labels: str, default=None
            .. deprecated::
                Use ``atlas`` instead. Accepts a path to a custom atlas CSV. If
                both ``atlas`` and ``atlas_labels`` are given, ``atlas`` wins.
                This parameter will be removed in a future release.
        device: str, default='cpu'
            Compute device: ``'cpu'``, or ``'cuda'``/``'gpu'`` to use an available
            GPU (falls back to CPU with a warning if CUDA is unavailable).
        random_state: int, default=42
            Seed for permutation generation. Uses a local RNG and does not modify
            the global NumPy/torch random state.
        """
        self.results_directory = results_directory

        # Convert string to TaskType enum if needed
        if isinstance(task_type, str):
            task_type = TaskType(task_type)
        self.task_type = task_type  # Will be validated/auto-detected in run()
        # Use a local seed for permutation generation instead of mutating the
        # global NumPy/torch RNG (which would silently affect the user's other code).
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
        self.calculate_residuals = calculate_residuals

        # The presence filter is meant to drop structural zeros from the raw
        # connectome; with global residualization the connectome is mean-centred
        # before selection, so the filter would see residualized values instead.
        if calculate_residuals and resolve_presence_threshold(
                getattr(self.edge_selection, 'presence_filter', False)) is not None:
            self.logger.warning(
                "Both calculate_residuals=True and a presence_filter are set: the "
                "presence filter will see residualized (not raw) connectome values."
            )
        self.n_permutations = n_permutations
        self.edge_significance_method = edge_significance_method
        self.nbs_threshold = nbs_threshold
        self.nbs_component_stat = nbs_component_stat

        if device.lower() == 'gpu' or device.lower() == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.logger.warning("CUDA or GPU not available, using CPU instead.")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

        # Log important configuration details
        self._log_analysis_details()

        # check inner cv and param grid
        if self.inner_cv is None:
            if len(self.edge_selection.param_grid) > 1:
                raise RuntimeError("Multiple hyperparameter configurations but no inner cv defined. "
                                   "Please provide only one hyperparameter configuration or an inner cv.")
            if self.select_stable_edges:
                raise RuntimeError("Stable edges can only be selected when using an inner cv.")

        # Resolve the atlas (built-in name or custom CSV path) to a DataFrame,
        # then persist it alongside the results so the report is self-contained.
        self.atlas = self._resolve_atlas(atlas, atlas_labels)
        self.atlas_labels = self._save_atlas(self.atlas)

        # results are saved to the results manager instance
        self.results_manager = None

    def _log_analysis_details(self):
        """
        Log important information about the analysis in a structured format.
        """
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
        self.logger.info(f"Calculate residuals:     {'Yes' if self.calculate_residuals else 'No'}")
        self.logger.info(f"Number of Permutations:  {self.n_permutations}")
        self.logger.info(f"Device:                  {self.device}")
        self.logger.info("="*50)

    def _resolve_atlas(self, atlas, atlas_labels):
        """
        Resolve the ``atlas`` / ``atlas_labels`` arguments to a validated
        DataFrame (or ``None``). ``atlas`` may be a built-in atlas name or a
        path to a custom CSV. ``atlas_labels`` is deprecated: it is accepted as
        a custom CSV path but ``atlas`` takes precedence when both are given.
        """
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
        """
        Persist the resolved atlas to ``<results_directory>/edges/atlas.csv`` so
        the HTML report is self-contained. Returns the saved path (or ``None``).
        """
        if atlas_df is None:
            return None
        dest_path = os.path.join(self.results_directory, "edges", "atlas.csv")
        atlas_df.to_csv(dest_path, index=False)
        self.logger.info(f"Saved atlas to {dest_path}")
        return dest_path

    def _validate_atlas_node_count(self, n_features):
        """
        Ensure the atlas region count matches the connectome node count implied
        by the number of edge features. A mismatched atlas would silently
        mislabel regions and misplace nodes in the brain plots.
        """
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

    def run(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, pd.DataFrame, np.ndarray],
            covariates: Union[pd.Series, pd.DataFrame, np.ndarray]):
        """
        Estimates a model using the provided data and conducts permutation testing. This method first fits the model to the actual data and subsequently performs estimation on permuted data for a specified number of permutations. Finally, it calculates permutation results.

        Parameters
        ----------
        X: Feature data used for the model. Can be a pandas DataFrame or a NumPy array.
        y: Target variable used in the estimation process. Can be a pandas Series, DataFrame, or a NumPy array.
        covariates: Additional covariate data to include in the model. Can be a pandas Series, DataFrame, or a NumPy array.

        """
        self.logger.info(f"Starting CPM estimation.")

        # check data and convert to numpy
        generate_data_insights(X=X, y=y, covariates=covariates, results_directory=self.results_directory)
        X, y, covariates = check_data(X, y, covariates, impute_missings=self.impute_missing_values)

        # Guard against an atlas that doesn't match the connectome size.
        self._validate_atlas_node_count(X.shape[1])

        # Detect or validate task type
        if self.task_type is None:
            self.task_type = detect_task_type(y)
            self.logger.info(f"Auto-detected task type: {self.task_type.value}")
        else:
            validate_task_type(y, self.task_type)
            self.logger.info(f"Using specified task type: {self.task_type.value}")

        # Save task type to results directory for HTML report
        with open(os.path.join(self.results_directory, 'task_type.txt'), 'w') as f:
            f.write(self.task_type.value)

        # Estimate models on actual data
        self._single_run(X=X, y=y.reshape(-1, 1), covariates=covariates, perm_run=False)
        self.logger.info("=" * 50)

        # Estimate models on permuted data
        if self.n_permutations > 0:
            self.logger.info(f"Running {self.n_permutations} permutations.")
            y_perms = self._create_permuted_y(y)
            self._single_run(X=X, y=y_perms, covariates=covariates, perm_run=True)
            PermutationManager.calculate_permutation_results(
                self.results_directory, self.logger,
                method=self.edge_significance_method,
                nbs_threshold=self.nbs_threshold,
                nbs_component_stat=self.nbs_component_stat)

        self.logger.info("=" * 50)
        self.logger.info("Estimation completed.")
        self.logger.info("Generating results file.")
        reporter = HTMLReporter(results_directory=self.results_directory, atlas_labels=self.atlas_labels)
        reporter.generate_html_report()

    def generate_html_report(self):
        self.logger.info("Generating HTML report.")
        reporter = HTMLReporter(results_directory=self.results_directory, atlas_labels=self.atlas_labels)
        reporter.generate_html_report()

    def _create_permuted_y(self, y):
        # 1. Create a matrix of the repeat vector
        y_tensor = torch.as_tensor(y, dtype=torch.float32)
        y_matrix = y_tensor.unsqueeze(0).expand(self.n_permutations, -1)

        # 2. Create random noise and get sorting indices (random permutation per row).
        # Use a local generator seeded from random_state so results are reproducible
        # without touching the global torch RNG.
        generator = torch.Generator(device=y_matrix.device).manual_seed(self.random_state)
        noise = torch.rand(y_matrix.shape, generator=generator,
                           dtype=y_matrix.dtype, device=y_matrix.device)
        indices = noise.argsort(dim=1)

        # 3. Apply these indices to permute each row
        permuted = y_matrix.gather(1, indices)

        # 4. Return as numpy [N_samples, N_perms] (boundary: this feeds into the pipeline)
        return permuted.t().numpy()

    def _single_run(self, X, y, covariates, perm_run: bool = False):
        """
        Perform a full cross-validation run (real data or permuted targets).

        Sets up the ResultsManager, iterates over outer folds, then
        aggregates and saves results.
        """
        if perm_run:
            results_directory = os.path.join(self.results_directory, "permutation")
        else:
            results_directory = self.results_directory

        # Retain per-fold edge masks (for edges.npy) only on the real run; the
        # permutation pass keeps just the fold-sum for the stability null, which
        # avoids a [Features, 2, Folds, n_permutations] tensor (huge for big
        # parcellations × many folds × many permutations).
        results_manager = ResultsManager(output_dir=results_directory, n_runs=y.shape[1],
                                         n_folds=self.cv.get_n_splits(), n_features=X.shape[1],
                                         device=self.device, store_fold_edges=not perm_run)

        # For a RepeatedKFold the outer split index runs 0..(n_splits*n_repeats-1)
        # with all folds of repeat 0 first, then repeat 1, etc. Derive the repeat
        # id so individual-level outputs can be averaged across repeats; plain
        # (non-repeated) CVs have n_repeats == 1, so every fold is repeat 0.
        n_repeats = getattr(self.cv, 'n_repeats', 1)
        splits_per_repeat = max(self.cv.get_n_splits() // n_repeats, 1)

        # The outer-fold loop can only be batched (multiple folds processed in
        # one torch call) when there's a single fixed edge-selection config --
        # with an inner CV, each fold's hyperparameter search is already
        # batched internally (see inner_fold.run_inner_folds) and owns its own
        # per-fold results directory, so the outer loop stays a plain
        # per-fold Python loop in that case. Non-linear models (no
        # fit_batched/predict_batched) always use the plain per-fold loop too.
        # calculate_residuals's train-fit/test-apply residualization is only
        # implemented in the per-fold loop (_run_outer_fold) -- rare enough
        # an option that a dedicated batched version isn't worth it, so fall
        # back to the loop rather than silently skip the residualization.
        # connected_components (networkx graph filtering, CPU-only, per fold)
        # and presence_filter both only run inside the per-fold edge-selection
        # path today -- presence_filter is applied in fit_transform_batched too,
        # but connected_components has no batched equivalent -- so exclude it
        # from batching the same way calculate_residuals is excluded below,
        # rather than silently skipping the filter under batching.
        can_batch_outer_folds = (
            self.inner_cv is None
            and not self.calculate_residuals
            and not self.edge_selection.connected_components
            and hasattr(self.cpm_model, 'fit_batched')
            and hasattr(self.cpm_model, 'predict_batched')
        )

        if can_batch_outer_folds:
            splits = list(self.cv.split(X, y[:, 0]))
            self._run_outer_folds_batched(splits, X, y, covariates, results_manager, perm_run,
                                          splits_per_repeat=splits_per_repeat)
        else:
            # Move the full dataset to the compute device ONCE, instead of
            # leaving X/y/covariates on the CPU for the whole outer-fold loop:
            # previously each of the (up to hundreds of) outer folds re-sliced
            # X on the CPU and every downstream torch.as_tensor(..., device=...)
            # call (inside run_inner_folds, model.fit, model.predict) re-uploaded
            # that fold's slice from scratch. With X/y/covariates already
            # device-resident here, torch_train_test_split's per-fold slicing
            # (and every downstream torch.as_tensor call, which is a no-op when
            # the input is already the right device/dtype) happens on-device
            # instead -- pure data placement, doesn't touch any statistic.
            # (self.cv.split still runs on the original CPU X/y: sklearn
            # splitters only derive index arrays from it, and some -- e.g.
            # StratifiedKFold -- aren't guaranteed to accept a CUDA tensor.)
            X_dev = torch.as_tensor(X, device=self.device, dtype=torch.float32)
            y_dev = torch.as_tensor(y, device=self.device, dtype=torch.float32)
            cov_dev = torch.as_tensor(covariates, device=self.device, dtype=torch.float32)

            iterator = tqdm(
                enumerate(self.cv.split(X, y[:, 0])),
                total=self.cv.get_n_splits(),
                desc="Running outer folds",
                unit="fold",
            )
            for outer_fold, (train, test) in iterator:
                repeat = outer_fold // splits_per_repeat
                self._run_outer_fold(outer_fold, repeat, train, test, X_dev, y_dev, cov_dev,
                                     results_manager, perm_run)

        # Aggregate across folds
        results_manager.calculate_final_cv_results(task_type=self.task_type)
        results_manager.calculate_edge_stability()

        if not perm_run:
            self.logger.info(results_manager.agg_results.round(4).to_string())
            results_manager.save_predictions()
            results_manager.save_network_strengths()
            self.results_manager = results_manager

    def _run_outer_fold(self, outer_fold, repeat, train, test, X, y, covariates,
                        results_manager, perm_run):
        with torch.cuda.nvtx.range("split_impute"):
            X_train, X_test, y_train, y_test, cov_train, cov_test = torch_train_test_split(
                train, test, X, y, covariates)
            if self.impute_missing_values:
                X_train, X_test, cov_train, cov_test = torch_impute_missing_values(
                    X_train, X_test, cov_train, cov_test)

        if self.calculate_residuals:
            with torch.cuda.nvtx.range("calculate_residuals"):
                X_train, X_test = residualize_train_test(X_train, X_test, cov_train, cov_test)

        with torch.cuda.nvtx.range("edge_selection"):
            edges = self._select_edges(X_train, y_train, cov_train,
                                       results_manager, outer_fold)
        with torch.cuda.nvtx.range("store_edges"):
            results_manager.store_edges(param_idx=0, fold_idx=outer_fold, edges_tensor=edges)

        with torch.cuda.nvtx.range("model_fit"):
            model = self.cpm_model(edges=edges, device=self.device, task_type=self.task_type)
            model.fit(X_train, y_train, cov_train)

        with torch.cuda.nvtx.range("model_predict"):
            y_pred = model.predict(X_test, cov_test, return_proba=True)

        if not perm_run:
            with torch.cuda.nvtx.range("store_predictions"):
                results_manager.store_predictions(y_pred=y_pred, y_true=y_test,
                                                  fold=outer_fold, test_indices=test,
                                                  repeat=repeat)
                network_strengths = model.get_network_strengths(X_test, cov_test)
                results_manager.store_network_strengths(network_strengths=network_strengths,
                                                        y_true=y_test, fold=outer_fold,
                                                        test_indices=test, repeat=repeat)

        with torch.cuda.nvtx.range("scoring"):
            metrics = score_models(y_true=y_test, y_pred=y_pred,
                                   task_type=self.task_type, device=self.device)

        with torch.cuda.nvtx.range("store_metrics"):
            results_manager.store_metrics(param_idx=0, fold_idx=outer_fold, metrics_tensor=metrics)

    def _run_outer_folds_batched(self, splits, X, y, covariates, results_manager, perm_run,
                                 splits_per_repeat=1):
        """
        Batched version of the outer-fold loop for the `inner_cv is None`
        case (a single fixed edge-selection config, no per-fold
        hyperparameter search): batches multiple outer folds -- and, memory
        permitting, permutation columns -- into single torch calls instead
        of one Python iteration per fold. Falls back to batch size 1
        (today's per-fold loop, routed through the same batched machinery
        with singleton batch dims) whenever memory is tight.
        """
        n_folds = len(splits)
        n_features = X.shape[1]
        n_perms = y.shape[1]
        n_cov = covariates.shape[1]

        X_gpu = torch.as_tensor(X, device=self.device, dtype=torch.float32)
        y_gpu = torch.as_tensor(y, device=self.device, dtype=torch.float32)
        cov_gpu = torch.as_tensor(covariates, device=self.device, dtype=torch.float32)

        n_samples_train = max(len(tr) for tr, te in splits)
        n_samples_test = max(len(te) for tr, te in splits)
        cost_fn = make_cpm_cost_fn(n_samples_train, n_samples_test, n_features, n_cov)
        avail = available_memory_bytes(self.device)
        with torch.cuda.nvtx.range("outer:plan_batch_sizes"):
            plan = plan_batch_sizes(1, n_folds, n_perms, cost_fn, avail)

        param_config = self.edge_selection.param_grid[0]
        selector = param_config['edge_selection']
        threshold = param_config['edge_selection__threshold']
        correction = param_config.get('edge_selection__correction')
        selector.correction = correction

        iterator = tqdm(range(0, n_folds, plan.folds), total=-(-n_folds // plan.folds),
                        desc="Running outer folds", unit="foldbatch")
        for fold_start in iterator:
            fold_ids = list(range(fold_start, min(fold_start + plan.folds, n_folds)))
            fold_splits = [splits[i] for i in fold_ids]
            fold_slice = slice(fold_ids[0], fold_ids[-1] + 1)

            with torch.cuda.nvtx.range(f"outer:foldbatch{fold_start}:split_impute"):
                fb = build_fold_batch(X_gpu, y_gpu, cov_gpu, fold_splits)
                if self.impute_missing_values:
                    fb.X_train, fb.X_test, fb.cov_train, fb.cov_test = torch_impute_missing_values_batched(
                        fb.X_train, fb.X_test, fb.cov_train, fb.cov_test, fb.train_valid)

            for perm_start in range(0, n_perms, plan.perms):
                perm_end = min(perm_start + plan.perms, n_perms)
                perm_slice = slice(perm_start, perm_end)
                y_train_p = fb.y_train[:, :, perm_slice]
                y_test_p = fb.y_test[:, :, perm_slice]

                with torch.cuda.nvtx.range(f"outer:foldbatch{fold_start}:permbatch{perm_start}:edge_selection"):
                    r_edges, p_edges = self.edge_selection.edge_statistic.fit_transform_batched(
                        X=fb.X_train, y=y_train_p, covariates=fb.cov_train,
                        valid_mask=fb.train_valid, device=self.device)
                    edges = selector.select_batch(r=r_edges, p=p_edges, thresholds=[threshold])  # [F,2,1,B,R]

                with torch.cuda.nvtx.range(f"outer:foldbatch{fold_start}:permbatch{perm_start}:store_edges"):
                    results_manager.store_edges(param_idx=0, fold_idx=fold_slice,
                                                edges_tensor=edges.squeeze(2), run_idx=perm_slice)

                with torch.cuda.nvtx.range(f"outer:foldbatch{fold_start}:permbatch{perm_start}:model_fit"):
                    model = self.cpm_model(edges=edges, device=self.device, task_type=self.task_type)
                    model.fit_batched(fb.X_train, y_train_p, fb.cov_train, valid_mask=fb.train_valid)

                with torch.cuda.nvtx.range(f"outer:foldbatch{fold_start}:permbatch{perm_start}:model_predict"):
                    y_pred = model.predict_batched(fb.X_test, fb.cov_test,
                                                    valid_mask=fb.test_valid, return_proba=True)  # [N,5,3,1,B,R]

                if not perm_run:
                    with torch.cuda.nvtx.range(
                            f"outer:foldbatch{fold_start}:permbatch{perm_start}:store_predictions"):
                        for b_local, outer_fold in enumerate(fold_ids):
                            train, test = splits[outer_fold]
                            n_test = len(test)
                            repeat = outer_fold // splits_per_repeat
                            results_manager.store_predictions(
                                y_pred=y_pred[:n_test, :, :, 0, b_local, :], y_true=fb.y_test[b_local, :n_test, perm_slice],
                                fold=outer_fold, test_indices=test, repeat=repeat)
                            # Network-strength reporting is a side-channel for the HTML
                            # report, not part of the hot batched path -- refit a plain
                            # single-fold model to reuse get_network_strengths() as-is.
                            edges_fold = edges[:, :, 0, b_local, :]
                            report_model = self.cpm_model(edges=edges_fold, device=self.device,
                                                          task_type=self.task_type)
                            report_model.fit(fb.X_train[b_local, :fb.n_train[b_local]],
                                             y_train_p[b_local, :fb.n_train[b_local]],
                                             fb.cov_train[b_local, :fb.n_train[b_local]])
                            network_strengths = report_model.get_network_strengths(
                                fb.X_test[b_local, :n_test], fb.cov_test[b_local, :n_test])
                            results_manager.store_network_strengths(
                                network_strengths=network_strengths,
                                y_true=fb.y_test[b_local, :n_test, perm_slice], fold=outer_fold,
                                test_indices=test, repeat=repeat)

                with torch.cuda.nvtx.range(f"outer:foldbatch{fold_start}:permbatch{perm_start}:scoring"):
                    metrics = score_models_batched(y_true=y_test_p, y_pred=y_pred, task_type=self.task_type,
                                                    valid_mask=fb.test_valid, device=self.device)

                with torch.cuda.nvtx.range(f"outer:foldbatch{fold_start}:permbatch{perm_start}:store_metrics"):
                    results_manager.store_metrics(param_idx=0, fold_idx=fold_slice,
                                                  metrics_tensor=metrics.squeeze(3), run_idx=perm_slice)

    def _select_edges(self, X_train, y_train, cov_train, results_manager, outer_fold):
        """
        Determine edge masks for this fold, either via inner CV hyperparameter
        search (with optional stability selection) or directly from the single
        configured edge-selection threshold.
        Returns
        -------
        edges : torch.Tensor [N_features, 2, N_runs]
        """
        if self.inner_cv:
            with torch.cuda.nvtx.range("edge_sel:inner_cv"):
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
        else:
            with torch.cuda.nvtx.range("edge_sel:param_setup"):
                best_params = [self.edge_selection.param_grid[0]] * y_train.shape[1]

        if self.select_stable_edges:
            with torch.cuda.nvtx.range("edge_sel:stable_edges"):
                return select_stable_edges(stability_edges, self.stability_threshold)

        with torch.cuda.nvtx.range("edge_sel:zeros_init"):
            edges = torch.zeros(X_train.shape[1], len(Networks) - 1, len(best_params),
                                device=self.device)

        torch.cuda.synchronize()
        with torch.cuda.nvtx.range("edge_sel:fit_transform"):
            r_edges, p_edges = self.edge_selection.edge_statistic.fit_transform(
                X=X_train, y=y_train, covariates=cov_train, device=self.device)
        torch.cuda.synchronize()

        if all(p == best_params[0] for p in best_params):
            with torch.cuda.nvtx.range("edge_sel:threshold_single"):
                self.edge_selection.set_params(**best_params[0])
                self.edge_selection.r_edges = r_edges
                self.edge_selection.p_edges = p_edges
                edges = self.edge_selection.return_selected_edges()
        else:
            for run_id, params in enumerate(best_params):
                with torch.cuda.nvtx.range(f"edge_sel:threshold_run{run_id}"):
                    self.edge_selection.set_params(**params)
                    self.edge_selection.r_edges = r_edges[:, [run_id]]
                    self.edge_selection.p_edges = p_edges[:, [run_id]]
                    current_edges = self.edge_selection.return_selected_edges()
                    edges[:, :, run_id] = current_edges.squeeze()

        return edges