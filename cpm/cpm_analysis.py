import os
import logging
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit, KFold

from cpm.fold import run_inner_folds
from cpm.logging import setup_logging
from cpm.models import LinearCPMModel
from cpm.edge_selection import UnivariateEdgeSelection, PThreshold
from cpm.results_manager import ResultsManager, PermutationManager
from cpm.utils import train_test_split, check_data, impute_missing_values, select_stable_edges
from cpm.scoring import score_regression_models


class CPMRegression:
    """
    This class handles the process of performing CPM Regression with cross-validation and permutation testing.
    """
    def __init__(self,
                 results_directory: str,
                 cv: Union[BaseCrossValidator, BaseShuffleSplit] = KFold(n_splits=10, shuffle=True, random_state=42),
                 inner_cv: Union[BaseCrossValidator, BaseShuffleSplit] = None,
                 edge_selection: UnivariateEdgeSelection = UnivariateEdgeSelection(
                     edge_statistic=['pearson'],
                     edge_selection=[PThreshold(threshold=[0.05], correction=[None])]
                 ),
                 select_stable_edges: bool = True,
                 stability_threshold: float = 0.8,
                 impute_missing_values: bool = True,
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
        self.cv = cv
        self.inner_cv = inner_cv
        self.edge_selection = edge_selection
        self.select_stable_edges = select_stable_edges
        self.stability_threshold = stability_threshold
        self.impute_missing_values = impute_missing_values
        self.n_permutations = n_permutations
        self.atlas_labels = atlas_labels

        np.random.seed(42)
        os.makedirs(self.results_directory, exist_ok=True)
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

    def _log_analysis_details(self):
        """
        Log important information about the analysis in a structured format.
        """
        self.logger.info("Starting CPM Regression Analysis")
        self.logger.info("="*50)
        self.logger.info(f"Results Directory:       {self.results_directory}")
        self.logger.info(f"Outer CV strategy:       {self.cv}")
        self.logger.info(f"Inner CV strategy:       {self.inner_cv}")
        self.logger.info(f"Edge selection method:   {self.edge_selection}")
        self.logger.info(f"Select stable edges:     {'Yes' if self.select_stable_edges else 'No'}")
        self.logger.info(f"Stability threshold:     {self.stability_threshold}")
        self.logger.info(f"Impute Missing Values:   {'Yes' if self.impute_missing_values else 'No'}")
        self.logger.info(f"Number of Permutations:  {self.n_permutations}")
        self.logger.info("="*50)

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
        X, y, covariates = check_data(X, y, covariates, impute_missings=self.impute_missing_values)

        # Estimate models on actual data
        self._single_run(X=X, y=y, covariates=covariates, perm_run=0)
        self.logger.info("=" * 50)

        # Estimate models on permuted data
        for perm_id in range(1, self.n_permutations + 1):
            self.logger.debug(f"Permutation run {perm_id}")
            y = np.random.permutation(y)
            self._single_run(X=X, y=y, covariates=covariates, perm_run=perm_id)

        if self.n_permutations > 0:
            PermutationManager.calculate_permutation_results(self.results_directory, self.logger)
        self.logger.info("Estimation completed.")

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

        for outer_fold, (train, test) in enumerate(self.cv.split(X, y)):
            if not perm_run:
                self.logger.debug(f"Running fold {outer_fold + 1}/{self.cv.get_n_splits()}")

            # split according to single outer fold
            X_train, X_test, y_train, y_test, cov_train, cov_test = train_test_split(train, test, X, y, covariates)

            # impute missing values
            if self.impute_missing_values:
                X_train, X_test, cov_train, cov_test = impute_missing_values(X_train, X_test, cov_train, cov_test)

            # if the user specified an inner cross-validation, estimate models witin inner loop
            if self.inner_cv:
                best_params, stability_edges = run_inner_folds(X=X_train, y=y_train, covariates=cov_train,
                                                               inner_cv=self.inner_cv,
                                                               edge_selection=self.edge_selection,
                                                               results_directory=os.path.join(results_manager.results_directory, 'folds', str(outer_fold)),
                                                               perm_run=perm_run)
                if not perm_run:
                    self.logger.info(f"Best hyperparameters: {best_params}")
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
            y_pred = LinearCPMModel(edges=edges).fit(X_train, y_train, cov_train).predict(X_test, cov_test)
            metrics = score_regression_models(y_true=y_test, y_pred=y_pred)
            results_manager.store_predictions(y_pred=y_pred, y_true=y_test, params=best_params, fold=outer_fold,
                                              param_id=0)
            results_manager.store_metrics(metrics=metrics, params=best_params, fold=outer_fold, param_id=0)

        # once all outer folds are done, calculate final results and edge stability
        results_manager.calculate_final_cv_results()
        results_manager.calculate_edge_stability()

        if not perm_run:
            self.logger.info(results_manager.agg_results.round(4).to_string())
            results_manager.save_predictions()
