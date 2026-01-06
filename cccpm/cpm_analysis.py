import os
import logging
import shutil

from typing import Union, Type
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit, KFold, RepeatedKFold, StratifiedKFold
from sklearn.linear_model import LinearRegression

from cccpm.fold import run_inner_folds
from cccpm.logging import setup_logging
from cccpm.more_models import BaseCPMModel, LinearCPMModel
from cccpm.edge_selection import UnivariateEdgeSelection, PThreshold
from cccpm.results_manager import ResultsManager, PermutationManager
from cccpm.utils import train_test_split, check_data, impute_missing_values, select_stable_edges, generate_data_insights
from cccpm.scoring import score_regression_models
from cccpm.reporting import HTMLReporter


class CPMRegression:
    """
    This class handles the process of performing CPM Regression with cross-validation and permutation testing.
    """
    def __init__(self,
                 results_directory: str,
                 cpm_model: Type[BaseCPMModel] = LinearCPMModel,
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
                 atlas_labels: str = None):
        """
        Initialize the CPMRegression object.

        Parameters
        ----------
        results_directory: str
            Directory to save results.
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
        self.cpm_model = cpm_model
        self.cv = cv
        self.inner_cv = inner_cv
        self.edge_selection = edge_selection
        self.select_stable_edges = select_stable_edges
        self.stability_threshold = stability_threshold
        self.impute_missing_values = impute_missing_values
        self.calculate_residuals = calculate_residuals
        self.n_permutations = n_permutations

        np.random.seed(42)
        os.makedirs(self.results_directory, exist_ok=True)
        os.makedirs(os.path.join(self.results_directory, "edges"), exist_ok=True)
        setup_logging(os.path.join(self.results_directory, "cpm_log.txt"))
        self.logger = logging.getLogger(__name__)

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
        self.logger.info("Starting CPM Regression Analysis")
        self.logger.info("="*50)
        self.logger.info(f"Results Directory:       {self.results_directory}")
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
        self.logger.info(f"Starting estimation with {self.n_permutations} permutations.")

        # check data and convert to numpy
        generate_data_insights(X=X, y=y, covariates=covariates, results_directory=self.results_directory)
        X, y, covariates = check_data(X, y, covariates, impute_missings=self.impute_missing_values)

        # Estimate models on actual data
        self._single_run(X=X, y=y, covariates=covariates, perm_run=0)
        self.logger.info("=" * 50)

        # Estimate models on permuted data
        for perm_id in tqdm(range(1, self.n_permutations + 1), desc="Permutation runs", unit="run",
                            total=self.n_permutations):
            y = np.random.permutation(y)
            self._single_run(X=X, y=y, covariates=covariates, perm_run=perm_id)

        if self.n_permutations > 0:
            PermutationManager.calculate_permutation_results(self.results_directory, self.logger)
        self.logger.info("Estimation completed.")
        self.logger.info("Generating results file.")
        reporter = HTMLReporter(results_directory=self.results_directory, atlas_labels=self.atlas_labels)
        reporter.generate_html_report()

    def generate_html_report(self):
        self.logger.info("Generating HTML report.")
        reporter = HTMLReporter(results_directory=self.results_directory, atlas_labels=self.atlas_labels)
        reporter.generate_html_report()

    def _single_run(self,
                    X: Union[pd.DataFrame, np.ndarray],
                    y: Union[pd.Series, pd.DataFrame, np.ndarray],
                    covariates: Union[pd.Series, pd.DataFrame, np.ndarray],
                    perm_run: int = 0):
        """
        Perform an estimation run (either real or permuted data). Includes outer cross-validation loop. For permutation
        runs, the same strategy is used, but printing is less verbose and the results folder changes.

        :param X: Features (predictors).
        :param y: Labels (target variable).
        :param covariates: Covariates to control for.
        :param perm_run: Permutation run identifier.
        """
        results_manager = ResultsManager(output_dir=self.results_directory, perm_run=perm_run,
                                         n_folds=self.cv.get_n_splits(), n_features=X.shape[1])

        iterator = (
            tqdm(
                enumerate(self.cv.split(X, y)),
                total=self.cv.get_n_splits(),
                desc="Running outer folds",
                unit="fold"
            )
            if not perm_run else
            enumerate(self.cv.split(X, y))
        )
        for outer_fold, (train, test) in iterator:
            # split according to single outer fold
            X_train, X_test, y_train, y_test, cov_train, cov_test = train_test_split(train, test, X, y, covariates)

            # impute missing values
            if self.impute_missing_values:
                X_train, X_test, cov_train, cov_test = impute_missing_values(X_train, X_test, cov_train, cov_test)

            # residualize X to remove effect of covariates
            if self.calculate_residuals:
                residual_model = LinearRegression().fit(cov_train, X_train)
                X_train = X_train - residual_model.predict(cov_train)
                X_test = X_test - residual_model.predict(cov_test)

            # if the user specified an inner cross-validation, estimate models witin inner loop
            if self.inner_cv:
                best_params, stability_edges = run_inner_folds(cpm_model=self.cpm_model,
                                                               X=X_train, y=y_train, covariates=cov_train,
                                                               inner_cv=self.inner_cv,
                                                               edge_selection=self.edge_selection,
                                                               results_directory=os.path.join(results_manager.results_directory, 'folds', str(outer_fold)),
                                                               perm_run=perm_run)
            else:
                best_params = self.edge_selection.param_grid[0]

            # Use best parameters to estimate performance on outer fold test set
            if self.select_stable_edges:
                edges = select_stable_edges(stability_edges, self.stability_threshold)
            else:
                self.edge_selection.set_params(**best_params)
                edges = self.edge_selection.fit_transform(X=X_train, y=y_train, covariates=cov_train).return_selected_edges()

            results_manager.store_edges(edges=edges, fold=outer_fold)

            # Build model and make predictions
            model = self.cpm_model(edges=edges).fit(X_train, y_train, cov_train)
            y_pred = model.predict(X_test, cov_test)
            network_strengths = model.get_network_strengths(X_test, cov_test)
            metrics = score_regression_models(y_true=y_test, y_pred=y_pred)
            results_manager.store_predictions(y_pred=y_pred, y_true=y_test, params=best_params, fold=outer_fold,
                                              param_id=0, test_indices=test)
            results_manager.store_metrics(metrics=metrics, params=best_params, fold=outer_fold, param_id=0)
            results_manager.store_network_strengths(network_strengths=network_strengths, y_true=y_test, fold=outer_fold)

        # once all outer folds are done, calculate final results and edge stability
        results_manager.calculate_final_cv_results()
        results_manager.calculate_edge_stability()

        if not perm_run:
            self.logger.info(results_manager.agg_results.round(4).to_string())
            results_manager.save_predictions()
            results_manager.save_network_strengths()
            self.results_manager = results_manager