import os
import logging
import shutil

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
from cccpm.edge_selection import UnivariateEdgeSelection, PThreshold
from cccpm.results_manager import ResultsManager, PermutationManager
from cccpm.utils import (train_test_split, check_data, impute_missing_values,
                         select_stable_edges, generate_data_insights, detect_task_type, validate_task_type)
from cccpm.scoring import score_models
from cccpm.reporting import HTMLReporter
from cccpm.constants import Networks, TaskType


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
                 atlas_labels: str = None,
                 device: str = 'cpu'):
        """
        Initialize the CPMRegression object.

        Parameters
        ----------
        results_directory: str
            Directory to save results.
        task_type: TaskType, str, or None
            Type of task: 'regression' or 'classification'.
            If None, will be auto-detected from target variable.
        cv: Union[BaseCrossValidator, BaseShuffleSplit]
            Outer cross-validation strategy.
        inner_cv: Union[BaseCrossValidator, BaseShuffleSplit]
            Inner cross-validation strategy for edge selection.
        edge_selection:  UnivariateEdgeSelection
            Method for edge selection.
        impute_missing_values: bool
            Whether to impute missing values.
        n_permutations: int
            Number of permutations to run for permutation testing.
        atlas_labels: str
            CSV file containing atlas and regions labels.
        """
        self.results_directory = results_directory

        # Convert string to TaskType enum if needed
        if isinstance(task_type, str):
            task_type = TaskType(task_type)
        self.task_type = task_type  # Will be validated/auto-detected in run()
        np.random.seed(42)
        torch.manual_seed(42)
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
        self.n_permutations = n_permutations

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

        # check and copy atlas labels file
        self.atlas_labels = self._validate_and_copy_atlas_file(atlas_labels)

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

    def _validate_and_copy_atlas_file(self, csv_path):
        """
        Validates that a CSV file exists and contains the required columns ('x', 'y', 'z', 'region').
        If valid, copies it to <self.results_directory>/edges.
        """
        if csv_path is None:
            return None

        required_columns = {"x", "y", "z", "region"}
        csv_path = os.path.abspath(csv_path)

        # Check if file exists
        if not os.path.isfile(csv_path):
            raise RuntimeError(f"CSV file does not exist: {csv_path}")

        # Try to read and validate columns
        try:
            df = pd.read_csv(csv_path)
            missing = required_columns - set(df.columns)

            if missing:
                raise RuntimeError(f"CSV file is missing required columns: {', '.join(missing)}")
        except Exception as e:
            raise RuntimeError(f"Error reading CSV file {csv_path}: {e}")

        # File and columns valid, proceed to copy
        dest_path = os.path.join(self.results_directory, "edges", os.path.basename(csv_path))

        try:
            shutil.copy(csv_path, dest_path)
            self.logger.info(f"Copied CSV file to {dest_path}")
            return dest_path
        except Exception as e:
            self.logger.error(f"Error copying file to {dest_path}: {e}")
            return None

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
            PermutationManager.calculate_permutation_results(self.results_directory, self.logger)

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

        # 2. Create random noise and get sorting indices (random permutation per row)
        noise = torch.rand_like(y_matrix)
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

        results_manager = ResultsManager(output_dir=results_directory, n_runs=y.shape[1],
                                         n_folds=self.cv.get_n_splits(), n_features=X.shape[1],
                                         device=self.device)

        iterator = tqdm(
            enumerate(self.cv.split(X, y[:, 0])),
            total=self.cv.get_n_splits(),
            desc="Running outer folds",
            unit="fold",
        )
        for outer_fold, (train, test) in iterator:
            self._run_outer_fold(outer_fold, train, test, X, y, covariates,
                                 results_manager, perm_run)

        # Aggregate across folds
        results_manager.calculate_final_cv_results(task_type=self.task_type)
        results_manager.calculate_edge_stability()

        if not perm_run:
            self.logger.info(results_manager.agg_results.round(4).to_string())
            results_manager.save_predictions()
            results_manager.save_network_strengths()
            self.results_manager = results_manager

    def _run_outer_fold(self, outer_fold, train, test, X, y, covariates,
                        results_manager, perm_run):
        """
        Execute a single outer CV fold: preprocess, select edges, fit model,
        predict, and store results.
        """
        # Split
        X_train, X_test, y_train, y_test, cov_train, cov_test = train_test_split(
            train, test, X, y, covariates)

        # Impute missing values
        if self.impute_missing_values:
            X_train, X_test, cov_train, cov_test = impute_missing_values(
                X_train, X_test, cov_train, cov_test)

        # Residualize X to remove effect of covariates
        if self.calculate_residuals:
            residual_model = LinearRegression().fit(cov_train, X_train)
            X_train = X_train - residual_model.predict(cov_train)
            X_test = X_test - residual_model.predict(cov_test)

        # Select edges (via inner CV or directly)
        edges = self._select_edges(X_train, y_train, cov_train,
                                   results_manager, outer_fold)
        results_manager.store_edges(param_idx=0, fold_idx=outer_fold, edges_tensor=edges)

        # Build model and make predictions
        model = self.cpm_model(edges=edges, device=self.device, task_type=self.task_type)
        model.fit(X_train, y_train, cov_train)
        y_pred = model.predict(X_test, cov_test, return_proba=True)

        if not perm_run:
            results_manager.store_predictions(y_pred=y_pred, y_true=y_test,
                                              fold=outer_fold, test_indices=test)
            network_strengths = model.get_network_strengths(X_test, cov_test)
            results_manager.store_network_strengths(network_strengths=network_strengths,
                                                    y_true=y_test, fold=outer_fold)

        # Score and store metrics
        metrics = score_models(y_true=y_test, y_pred=y_pred,
                               task_type=self.task_type, device=self.device)
        results_manager.store_metrics(param_idx=0, fold_idx=outer_fold, metrics_tensor=metrics)

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
        else:
            best_params = [self.edge_selection.param_grid[0]] * y_train.shape[1]

        if self.select_stable_edges:
            return select_stable_edges(stability_edges, self.stability_threshold)

        edges = torch.zeros(X_train.shape[1], len(Networks) - 1, len(best_params))
        for run_id, params in enumerate(best_params):
            self.edge_selection.set_params(**params)
            current_edges = self.edge_selection.fit_transform(
                X=X_train, y=y_train[:, run_id].reshape(-1, 1), covariates=cov_train
            ).return_selected_edges()
            edges[:, :, run_id] = current_edges.squeeze()

        return edges