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
from cccpm.preprocessing import (torch_train_test_split, torch_impute_missing_values,
                                 residualize_train_test, select_stable_edges)
from cccpm.validation import (check_data, detect_task_type, validate_task_type,
                              infer_n_nodes)
from cccpm.memory import plan_permutation_chunk
from cccpm.atlases import resolve_atlas
from cccpm.scoring import score_models
from cccpm.reporting import HTMLReporter
from cccpm.reporting.data_insights import generate_data_insights
from cccpm.constants import Models, Networks, TaskType


# Parameters renamed in 0.7.0, mapped to their replacement.
#
# "edge significance" conflated two different questions: the p-value that decides
# whether an edge is *selected* (set on PThreshold), and whether an edge is
# selected across folds *more consistently than chance*. These parameters only
# ever meant the second -- which is what the outputs have always been called
# (stability_edges_significance.npy). `nbs_threshold` had the same problem one
# level down: it is a fraction of folds, not a p-value, and it sat in the same
# call as `PThreshold(threshold=...)` with nothing to tell them apart.
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
        model_input: str, default='raw'
            What the predictive models consume. ``'raw'`` uses the connectome as
            given; ``'residualized'`` regresses the covariates out of it first,
            fitting the residualiser on each training split and applying it to the
            held-out split. Independent of ``selection_input``, which controls edge
            selection: together they are the 2x2 of confound control.

            This is a property of the run, not an extra model. It has to be: OLS
            is invariant to it once the covariates are in the design -- `full` and
            `increment` are unchanged -- but a tree, forest or GAM is not. On
            identical edges, fitting `full` on raw versus residualised connectivity
            moves predictions by 391% of sd(y) for ``DecisionTreeCPM``, 65% for
            ``RandomForestCPM`` and 19% for ``GAMCPM``, against 0.0% for
            ``LinearCPM``. A "residualised" *model* would therefore mean something
            different for every backend, and the user could not tell from the
            results which connectome produced `full`.
        n_permutations: int, default=0
            Number of label permutations for significance testing. ``0`` disables
            permutation testing; use 1000+ for publishable p-values.
        stability_significance_method: str, default='nbs'
            How edge-*stability* significance is established from the
            permutations -- that is, whether an edge is selected across folds
            more consistently than chance. This is a different question from
            which edges pass the selection threshold in the first place, which
            is set on :class:`PThreshold`. ``'nbs'`` uses the Network-Based
            Statistic (connected-component test, subnetwork-level FWER control);
            ``'tfce'`` uses network Threshold-Free Cluster Enhancement (per-edge
            FWER control, no primary threshold).
        nbs_stability_threshold: float, default=0.5
            **Stability** threshold (``>=``) for NBS component forming -- a
            fraction of folds, not a p-value. Because stability is discrete over
            the outer folds, ``0.5`` keeps edges selected in a majority of them.
            Ignored when the method is ``'tfce'``.
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
        self.logger.info(f"Selection input:         {self.edge_selection.statistic._input}")
        self.logger.info(f"Model input:             {self.model_input}")
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

    def _available_models(self):
        """
        Which model variants this run defines. Without covariates only
        ``connectome`` is meaningful: ``covariates`` has an empty design,
        ``full`` collapses onto ``connectome``, and ``increment`` would be
        identically zero.
        """
        if getattr(self, 'has_covariates', True):
            return [m.name for m in Models]
        return [Models.connectome.name]

    def _validate_covariate_requirements(self, covariates):
        """
        Fail up front when an option that needs covariates was combined with no
        covariates, naming the offending parameter.

        Without this the run would not crash -- it would quietly degrade. A
        ``*_partial`` statistic with an empty confound design is just the plain
        statistic with an empty confound design is just the plain statistic, so
        the run would produce a full set of plausible numbers that silently
        answer a different question -- the failure mode this package has been
        bitten by before.
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
        """
        Estimates a model using the provided data and conducts permutation testing. This method first fits the model to the actual data and subsequently performs estimation on permuted data for a specified number of permutations. Finally, it calculates permutation results.

        Parameters
        ----------
        X: Feature data used for the model. Can be a pandas DataFrame or a NumPy array.
        y: Target variable used in the estimation process. Can be a pandas Series, DataFrame, or a NumPy array.
        covariates: Additional covariate data to include in the model. Can be a pandas Series, DataFrame, or a NumPy array.
            Omit it (or pass ``None``) for vanilla CPM with no confound control. In that
            mode only the ``connectome`` model is defined -- ``covariates``, ``full``,
            ``full`` and ``increment`` need covariates and are reported as NaN,
            and the models that do exist are listed in ``available_models.json``.
            Options that presuppose covariates (a ``*_partial`` edge statistic,
            ``model_input='residualized'``) then raise up front.

        """
        self.logger.info("Starting CPM estimation.")

        self._validate_covariate_requirements(covariates)

        # check data and convert to numpy
        generate_data_insights(X=X, y=y, covariates=covariates, results_directory=self.results_directory)
        X, y, covariates = check_data(X, y, covariates, impute_missings=self.impute_missing_values)

        self.has_covariates = covariates is not None
        if not self.has_covariates:
            self.logger.info(
                "No covariates supplied: running vanilla CPM. Only the "
                "'connectome' model is defined; covariates/full/"
                "increment are reported as NaN.")
            # A zero-width design keeps every tensor operation downstream valid
            # without a `None` check in each of them. LinearCPM reads the width,
            # not this flag, to decide which variants exist.
            covariates = np.empty((X.shape[0], 0), dtype=np.float64)

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

        # Same, for the confound configuration. The report has to be able to say
        # which cell of the selection_input x model_input 2x2 produced it --
        # otherwise a naive run and a fully controlled one are indistinguishable
        # to anyone who is handed the HTML. Written as a file rather than parsed
        # back out of the log, so it survives a run with logging turned down.
        with open(os.path.join(self.results_directory, 'run_config.json'), 'w') as f:
            json.dump({
                'selection_statistic': self.edge_selection.statistic._statistic,
                'selection_input': self.edge_selection.statistic._input,
                'model_input': self.model_input,
                'has_covariates': self.has_covariates,
            }, f)

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
                method=self.stability_significance_method,
                nbs_stability_threshold=self.nbs_stability_threshold,
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

        Sets up the ResultsManager, iterates over outer folds, then aggregates
        and saves results.
        """
        if perm_run:
            results_directory = os.path.join(self.results_directory, "permutation")
        else:
            results_directory = self.results_directory

        # Retain per-fold edge masks (for edges.npy) only on the real run; the
        # permutation pass keeps just the fold-sum for the stability null, which
        # avoids a [Features, 2, Folds, n_permutations] tensor (huge for big
        # parcellations x many folds x many permutations).
        results_manager = ResultsManager(output_dir=results_directory, n_runs=y.shape[1],
                                         n_folds=self.cv.get_n_splits(), n_features=X.shape[1],
                                         device=self.device, store_fold_edges=not perm_run,
                                         available_models=self._available_models())

        # For a RepeatedKFold the outer split index runs 0..(n_splits*n_repeats-1)
        # with all folds of repeat 0 first, then repeat 1, etc. Derive the repeat
        # id so individual-level outputs can be averaged across repeats; plain
        # (non-repeated) CVs have n_repeats == 1, so every fold is repeat 0.
        n_repeats = getattr(self.cv, 'n_repeats', 1)
        splits_per_repeat = max(self.cv.get_n_splits() // n_repeats, 1)

        # Move the full dataset to the compute device once, rather than
        # re-uploading each fold's slice on every downstream torch.as_tensor
        # call. Pure data placement; touches no statistic. (self.cv.split still
        # runs on the original CPU X/y: sklearn splitters only derive index
        # arrays from it, and some -- e.g. StratifiedKFold -- are not guaranteed
        # to accept a CUDA tensor.)
        X_dev = torch.as_tensor(X, device=self.device, dtype=torch.float32)
        y_dev = torch.as_tensor(y, device=self.device, dtype=torch.float32)
        cov_dev = torch.as_tensor(covariates, device=self.device, dtype=torch.float32)

        # Permutations stay fully vectorised; this only caps how many columns are
        # in flight so a large parcellation x many permutations degrades in speed
        # rather than running out of memory. For a real (non-permutation) run
        # n_runs == 1, so it never engages.
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

        # Aggregate across folds
        results_manager.calculate_final_cv_results(task_type=self.task_type)
        results_manager.calculate_edge_stability()

        if not perm_run:
            self.logger.info(results_manager.agg_results.round(4).to_string())
            results_manager.save_predictions()
            results_manager.save_network_strengths()
            self.results_manager = results_manager

    def _run_outer_fold(self, outer_fold, repeat, train, test, X, y, covariates,
                        results_manager, perm_run, run_idx=slice(None)):
        X_train, X_test, y_train, y_test, cov_train, cov_test = torch_train_test_split(
            train, test, X, y, covariates)
        if self.impute_missing_values:
            X_train, X_test, cov_train, cov_test = torch_impute_missing_values(
                X_train, X_test, cov_train, cov_test)

        # edges: [Features, 2, Runs] -- one winning configuration per run.
        # Selection always sees the connectome as supplied; whether it controls
        # for the covariates is decided inside the statistic by selection_input.
        edges = self._select_edges(X_train, y_train, cov_train, results_manager, outer_fold)
        results_manager.store_edges(param_idx=0, fold_idx=outer_fold, edges_tensor=edges,
                                    run_idx=run_idx)

        # Deconfound the features the models consume, if asked. Fitted on train
        # and applied to test, and deliberately *after* selection so the two
        # choices stay independent.
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
        """
        Determine edge masks for this fold, either via inner CV hyperparameter
        search (with optional stability selection) or directly from the single
        configured edge-selection threshold.

        Returns
        -------
        edges : torch.Tensor [N_features, 2, N_runs]
        """
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

        r_edges, p_edges = self.edge_selection.statistic.fit_transform(
            X=X_train, y=y_train, covariates=cov_train, device=self.device)
        self.edge_selection.r_edges = r_edges
        self.edge_selection.p_edges = p_edges

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
            self.edge_selection.p_edges = p_edges[:, [run_id]]
            edges[:, :, run_id] = self.edge_selection.return_selected_edges()[:, :, 0, 0]
        return edges
