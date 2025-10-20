# import os
# import logging
# import shutil
#
# from typing import Union
#
# import numpy as np
# import torch
# import pandas as pd
# from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit, KFold
#
# import cpm.scoring
# from cpm.fold import run_inner_folds
# from cpm.logging import setup_logging
# from cpm.models import LinearCPMModel
# from cpm.edge_selection import UnivariateEdgeSelection, PThreshold
# from cpm.results_manager import ResultsManager, PermutationManager
# from cpm.utils import train_test_split, check_data, impute_missing_values, select_stable_edges, generate_data_insights
# from cpm.scoring import score_regression_models
# from cpm.reporting import HTMLReporter
#
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
#
#
# class CPMRegression:
#     """
#     This class handles the process of performing CPM Regression with cross-validation and permutation testing.
#     """
#     def __init__(self,
#                  results_directory: str,
#                  cv: Union[BaseCrossValidator, BaseShuffleSplit] = KFold(n_splits=10, shuffle=True, random_state=42),
#                  inner_cv: Union[BaseCrossValidator, BaseShuffleSplit] = None,
#                  edge_selection: UnivariateEdgeSelection = UnivariateEdgeSelection(
#                      edge_statistic='pearson',
#                      edge_selection=[PThreshold(threshold=[0.05], correction=[None])]
#                  ),
#                  select_stable_edges: bool = False,
#                  stability_threshold: float = 0.8,
#                  impute_missing_values: bool = True,
#                  n_permutations: int = 0,
#                  atlas_labels: str = None,
#                  lambda_reg: float = 1e-5):
#         """
#         Initialize the CPMRegression object.
#
#         Parameters
#         ----------
#         results_directory: str
#             Directory to save results.
#         cv: Union[BaseCrossValidator, BaseShuffleSplit]
#             Outer cross-validation strategy.
#         inner_cv: Union[BaseCrossValidator, BaseShuffleSplit]
#             Inner cross-validation strategy for edge selection.
#         edge_selection:  UnivariateEdgeSelection
#             Method for edge selection.
#         impute_missing_values: bool
#             Whether to impute missing values.
#         n_permutations: int
#             Number of permutations to run for permutation testing.
#         atlas_labels: str
#             CSV file containing atlas and regions labels.
#         """
#         self.lambda_reg = lambda_reg
#         self.results_directory = results_directory
#         self.cv = cv
#         self.inner_cv = inner_cv
#         self.edge_selection = edge_selection
#         self.select_stable_edges = select_stable_edges
#         self.stability_threshold = stability_threshold
#         self.impute_missing_values = impute_missing_values
#         self.n_permutations = n_permutations
#
#         print(edge_selection)
#
#         np.random.seed(42)
#         os.makedirs(self.results_directory, exist_ok=True)
#         os.makedirs(os.path.join(self.results_directory, "edges"), exist_ok=True)
#         setup_logging(os.path.join(self.results_directory, "cpm_log.txt"))
#         self.logger = logging.getLogger(__name__)
#
#         # Log important configuration details
#         self._log_analysis_details()
#
#         # check inner cv and param grid
#         if self.inner_cv is None:
#             if len(self.edge_selection.param_grid) > 1:
#                 raise RuntimeError("Multiple hyperparameter configurations but no inner cv defined. "
#                                    "Please provide only one hyperparameter configuration or an inner cv.")
#             if self.select_stable_edges:
#                 raise RuntimeError("Stable edges can only be selected when using an inner cv.")
#
#         # check and copy atlas labels file
#         self.atlas_labels = self._validate_and_copy_atlas_file(atlas_labels)
#
#     def _log_analysis_details(self):
#         """
#         Log important information about the analysis in a structured format.
#         """
#         self.logger.info("Starting CPM Regression Analysis")
#         self.logger.info("="*50)
#         self.logger.info(f"Results Directory:       {self.results_directory}")
#         self.logger.info(f"Outer CV strategy:       {self.cv}")
#         self.logger.info(f"Inner CV strategy:       {self.inner_cv}")
#         self.logger.info(f"Edge selection method:   {self.edge_selection}")
#         self.logger.info(f"Select stable edges:     {'Yes' if self.select_stable_edges else 'No'}")
#         if self.select_stable_edges:
#             self.logger.info(f"Stability threshold:     {self.stability_threshold}")
#         self.logger.info(f"Impute Missing Values:   {'Yes' if self.impute_missing_values else 'No'}")
#         self.logger.info(f"Number of Permutations:  {self.n_permutations}")
#         self.logger.info("="*50)
#
#     def _validate_and_copy_atlas_file(self, csv_path):
#         """
#         Validates that a CSV file exists and contains the required columns ('x', 'y', 'z', 'region').
#         If valid, copies it to <self.results_directory>/edges.
#         """
#         if csv_path is None:
#             return None
#
#         required_columns = {"x", "y", "z", "region"}
#         csv_path = os.path.abspath(csv_path)
#
#         # Check if file exists
#         if not os.path.isfile(csv_path):
#             raise RuntimeError(f"CSV file does not exist: {csv_path}")
#
#         # Try to read and validate columns
#         try:
#             df = pd.read_csv(csv_path)
#             missing = required_columns - set(df.columns)
#
#             if missing:
#                 raise RuntimeError(f"CSV file is missing required columns: {', '.join(missing)}")
#         except Exception as e:
#             raise RuntimeError(f"Error reading CSV file {csv_path}: {e}")
#
#         # File and columns valid, proceed to copy
#         dest_path = os.path.join(self.results_directory, "edges", os.path.basename(csv_path))
#
#         try:
#             shutil.copy(csv_path, dest_path)
#             self.logger.info(f"Copied CSV file to {dest_path}")
#             return dest_path
#         except Exception as e:
#             self.logger.error(f"Error copying file to {dest_path}: {e}")
#             return None
#
#     def run(self,
#             X: Union[pd.DataFrame, np.ndarray],
#             y: Union[pd.Series, pd.DataFrame, np.ndarray],
#             covariates: Union[pd.Series, pd.DataFrame, np.ndarray]):
#         """
#         Estimates a model using the provided data and conducts permutation testing. This method first fits the model to the actual data and subsequently performs estimation on permuted data for a specified number of permutations. Finally, it calculates permutation results.
#
#         Parameters
#         ----------
#         X: Feature data used for the model. Can be a pandas DataFrame or a NumPy array.
#         y: Target variable used in the estimation process. Can be a pandas Series, DataFrame, or a NumPy array.
#         covariates: Additional covariate data to include in the model. Can be a pandas Series, DataFrame, or a NumPy array.
#
#         """
#         self.logger.info(f"Starting estimation with {self.n_permutations} permutations.")
#
#         generate_data_insights(X=X, y=y, covariates=covariates, results_directory=self.results_directory)
#         X, y, covariates = check_data(X, y, covariates, impute_missings=self.impute_missing_values)
#
#
#         self.logger.info("=" * 50)
#
#         # Estimate models on permuted data
#         p = self.n_permutations + 1
#         cov_tensor = covariates
#         cov_tot = cov_tensor.unsqueeze(0).repeat(p, 1, 1)  # (p, n_samples, n_cov)
#
#         y_tot = torch.zeros((p, y.shape[0]), dtype=y.dtype)
#         X_tot = torch.zeros((p, *X.shape), dtype=X.dtype)
#
#         y_tot[0] = y
#         X_tot[0] = X
#
#         for i in range(1, p):
#             idx = torch.randperm(y.shape[0])
#             y_tot[i] = y[idx]
#             X_tot[i] = X[idx]
#
#         X_tot.to(device)
#         y_tot.to(device)
#         cov_tot.to(device)
#
#
#         import time
#         start = time.time()
#         results = self._batch_run(X=X_tot, y=y_tot, covariates=cov_tot)
#         end = time.time()
#         self.logger.info(f"Estimation took {end-start:.2f} seconds")
#
#         results_managers = []
#         for b in range(len(results["metrics_detailed"])):
#             rm = ResultsManager(output_dir=self.results_directory, perm_run=b,
#                                 n_folds=self.cv.get_n_splits(), n_features=X.shape[-1])
#             results_managers.append(rm)
#
#         for b, results_manager in enumerate(results_managers):
#             for fold in range(self.cv.get_n_splits()):
#                 edge_mask = results["edge_masks"][b, fold].cpu().numpy()
#
#                 edges = {
#                     'positive': np.where(edge_mask > 0)[0],
#                     'negative': np.array([], dtype=int)
#                 }
#
#                 fold_key = f'fold_{fold}'
#                 scores = results["metrics_detailed"][b][fold_key]
#
#                 results_manager.store_edges(edges=edges, fold=fold)
#                 results_manager.store_metrics(metrics=results["metrics_detailed"][b][f'fold_{fold}'], params={}, fold=fold, param_id=0)
#
#                 y_pred = results["predictions"][b][fold_key]
#                 y_true = y_tot[b][fold * (y.shape[0] // self.cv.get_n_splits()):(fold + 1) * (
#                             y.shape[0] // self.cv.get_n_splits())].cpu().numpy()
#                 test_idx = np.arange(y_true.shape[0])
#                 results_manager.store_predictions(
#                     y_pred=y_pred,
#                     y_true=y_true,
#                     params={},
#                     fold=fold,
#                     param_id=0,
#                     test_indices=test_idx
#                 )
#
#                 network_strengths = results["network_strengths"][b][fold_key]
#                 y_true = y_tot[b][fold * (y.shape[0] // self.cv.get_n_splits()):
#                                   (fold + 1) * (y.shape[0] // self.cv.get_n_splits())].cpu().numpy()
#
#                 results_manager.store_network_strengths(
#                     network_strengths=network_strengths,
#                     y_true=y_true,
#                     fold=fold
#                 )
#
#             results_manager.calculate_final_cv_results()
#             results_manager.calculate_edge_stability()
#
#             results_manager.save_predictions()
#             results_manager.save_network_strengths()
#
#             self.logger.info(results_manager.agg_results.round(4).to_string())
#
#         # reporter = HTMLReporter(results_directory=self.results_directory, atlas_labels=self.atlas_labels)
#         # reporter.generate_html_report()
#
#         if self.n_permutations > 0:
#             PermutationManager.calculate_permutation_results(self.results_directory, self.logger)
#         self.logger.info("Estimation completed.")
#         self.logger.info("Generating results file.")
#         reporter = HTMLReporter(results_directory=self.results_directory, atlas_labels=self.atlas_labels)
#         reporter.generate_html_report()
#
#     # def generate_html_report(self):
#     #     self.logger.info("Generating HTML report.")
#     #     reporter = HTMLReporter(results_directory=self.results_directory, atlas_labels=self.atlas_labels)
#     #     reporter.generate_html_report()
#
#     def _batch_run(self,
#                    X: torch.Tensor,
#                    y: torch.Tensor,
#                    covariates: torch.Tensor,
#                    ) -> dict:
#         B, n, p = X.shape
#         _, _, c = covariates.shape
#         n_folds = self.cv.get_n_splits()
#
#         edge_masks_all = torch.zeros(B, n_folds, p, device=device)
#         metrics_all = [{} for _ in range(B)]  # pro Batch ein Metrik-Dict wie score_regression_models
#         all_predictions_dict = [{} for _ in range(B)]  # [batch][fold] -> y_pred_dict
#         all_network_strengths = [{} for _ in range(B)]  # [batch][fold] -> network_strengths
#
#         for outer_fold, (train_idx, test_idx) in enumerate(self.cv.split(torch.arange(n))):
#             # Split
#             Xtr, Xte = X[:, train_idx].to(device), X[:, test_idx].to(device)
#             ytr, yte = y[:, train_idx].to(device), y[:, test_idx].to(device)
#             covtr, covte = covariates[:, train_idx].to(device), covariates[:, test_idx].to(device)
#
#             # Edge Selection
#             r_edges, p_edges = self.edge_selection.fit_transform(Xtr, ytr, covtr)  # [B, p]
#             edge_mask = (p_edges < 0.01).to(dtype=torch.bool).squeeze()  # [B, p]
#             edge_masks_all[:, outer_fold] = edge_mask
#
#             models = [LinearCPMModel(device=device) for _ in range(B)]
#
#             # Vectorized training + prediction (we need to move model creation out of vmap long term)
#             def train_and_predict(model, Xtr_b, Xte_b, covtr_b, covte_b, ytr_b, yte_b, edge_mask_b):
#                 model.fit(Xtr_b, ytr_b, covtr_b, edge_mask_b)  # Fit the pre-created model
#                 preds = model.predict(Xte_b, covte_b, edge_mask_b)
#                 strengths = model.get_network_strengths(Xte_b, covte_b, edge_mask_b)
#                 return preds, strengths
#
#             batched_fn = torch.vmap(
#                 train_and_predict,
#                 in_dims=(0, 0, 0, 0, 0, 0, 0),  # batch dim on all inputs
#                 randomness="different"
#             )
#             models_tuple = tuple(models)
#             preds_all, strengths_all = batched_fn(
#                 models_tuple, Xtr, Xte, covtr, covte, ytr, yte, edge_mask
#             )
#
#             # Now compute metrics per batch using CPU loop
#             for b in range(B):
#                 scores = score_regression_models(
#                     y_true=yte[b],
#                     y_pred_tensor=preds_all[b],  # you’ll need to adapt scorer to accept tensor
#                     primary_metric_only=False
#                 )
#                 metrics_all[b][f"fold_{outer_fold}"] = scores
#                 all_predictions_dict[b][f"fold_{outer_fold}"] = preds_all[b]
#                 all_network_strengths[b][f"fold_{outer_fold}"] = strengths_all[b]
#
#         return {
#             "edge_masks": edge_masks_all,  # [B, n_folds, p]
#             "metrics_detailed": metrics_all,
#             "predictions": all_predictions_dict,
#             "network_strengths": all_network_strengths
#         }
#
#         #     for b in range(B):
#         #         print(edge_mask.shape)
#         #         mask_pos = edge_mask[b, :]
#         #         print("mask_pos shape:", mask_pos.shape)  # Sollte [p] sein
#         #         print("Xtr[b] shape:", Xtr[b].shape)
#         #         Xb_tr, Xb_te = Xtr[b][:, mask_pos], Xte[b][:, mask_pos]
#         #         cov_b_tr, cov_b_te = covtr[b], covte[b]
#         #         yb_tr = ytr[b].unsqueeze(1)
#         #         yb_te = yte[b]
#         #
#         #         # Modellvarianten: CONNECTOME ONLY
#         #         X_connectome_train = Xb_tr
#         #         X_connectome_test = Xb_te
#         #
#         #         # COVARIATES ONLY
#         #         X_cov_train = cov_b_tr
#         #         X_cov_test = cov_b_te
#         #
#         #         # FULL MODEL
#         #         X_full_train = torch.cat([Xb_tr, cov_b_tr], dim=1)
#         #         X_full_test = torch.cat([Xb_te, cov_b_te], dim=1)
#         #
#         #         cov_b_tr = cov_b_tr.float()
#         #         yb_tr = yb_tr.float()
#         #
#         #         # RESIDUALS: erst Covariate-Modell, dann Residuen
#         #         beta_cov = torch.linalg.solve(
#         #             cov_b_tr.T @ cov_b_tr + self.lambda_reg * torch.eye(c, device=device),
#         #             cov_b_tr.T @ yb_tr
#         #         ).squeeze()
#         #         y_cov_pred = (cov_b_te @ beta_cov).squeeze()
#         #         y_residual = yb_te - y_cov_pred  # Ziel für residual model
#         #
#         #         # Modellschätzungen
#         #         def osl(X_train, X_test, y_train): # linear regression closed form
#         #             XTX = X_train.T @ X_train + self.lambda_reg * torch.eye(X_train.shape[1], device=device)
#         #             XTy = X_train.T @ y_train
#         #             beta = torch.linalg.solve(XTX, XTy).squeeze()
#         #             return (X_test @ beta).squeeze()
#         #
#         #         y_pred_dict = {
#         #             'connectome': {
#         #                 'positive': osl(X_connectome_train, X_connectome_test, yb_tr),
#         #                 'negative': torch.zeros_like(yb_te),  # Nur positiv verwendet
#         #                 'both': osl(X_connectome_train, X_connectome_test, yb_tr)
#         #             },
#         #             'covariates': {
#         #                 'positive': osl(X_cov_train, X_cov_test, yb_tr),
#         #                 'negative': torch.zeros_like(yb_te),
#         #                 'both': osl(X_cov_train, X_cov_test, yb_tr)
#         #             },
#         #             'full': {
#         #                 'positive': osl(X_full_train, X_full_test, yb_tr),
#         #                 'negative': torch.zeros_like(yb_te),
#         #                 'both': osl(X_full_train, X_full_test, yb_tr)
#         #             },
#         #             'residuals': {
#         #                 'positive': osl(X_connectome_train, X_connectome_test, y_residual.unsqueeze(1)),
#         #                 'negative': torch.zeros_like(yb_te),
#         #                 'both': osl(X_connectome_train, X_connectome_test, y_residual.unsqueeze(1))
#         #             }
#         #         }
#         #
#         #         # Metriken berechnen und sammeln
#         #         scores = score_regression_models(y_true=yb_te, y_pred_dict=y_pred_dict, primary_metric_only=False)
#         #         metrics_all[b][f'fold_{outer_fold}'] = scores
#         #
#         # return {
#         #     "edge_masks": edge_masks_all,  # [B, n_folds, p]
#         #     "metrics_detailed": metrics_all  # Liste von Dictionairies mit Struktur wie score_regression_models
#         # }


import os
import logging
import shutil
import gc

from typing import Union

import math
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import psutil
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit, KFold

import cpm.scoring
from cpm.fold import run_inner_folds
from cpm.logging import setup_logging
from cpm.models import LinearCPMModel
from cpm.edge_selection import UnivariateEdgeSelection, PThreshold, EdgeStatistic
from cpm.results_manager import ResultsManager, PermutationManager
from cpm.utils import train_test_split, check_data, impute_missing_values, select_stable_edges, generate_data_insights
from cpm.scoring import score_regression_models
from cpm.reporting import HTMLReporter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)


class CPMRegression:
    """
    This class handles the process of performing CPM Regression with cross-validation and permutation testing.
    """

    def __init__(self,
                 results_directory: str,
                 cv: Union[BaseCrossValidator, BaseShuffleSplit] = KFold(n_splits=10, shuffle=True, random_state=42),
                 inner_cv: Union[BaseCrossValidator, BaseShuffleSplit] = None,
                 edge_selection: UnivariateEdgeSelection = UnivariateEdgeSelection(
                     edge_statistic='pearson',
                     edge_selection=[PThreshold(threshold=[0.05], correction=[None])]
                 ),
                 select_stable_edges: bool = False,
                 stability_threshold: float = 0.8,
                 impute_missing_values: bool = True,
                 n_permutations: int = 0,
                 atlas_labels: str = None,
                 lambda_reg: float = 1e-5):
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
        self.lambda_reg = lambda_reg
        self.results_directory = results_directory
        self.cv = cv
        self.inner_cv = inner_cv
        self.edge_selection = edge_selection
        self.select_stable_edges = select_stable_edges
        self.stability_threshold = stability_threshold
        self.impute_missing_values = impute_missing_values
        self.n_permutations = n_permutations

        #print(edge_selection)

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

    def _log_analysis_details(self):
        """
        Log important information about the analysis in a structured format.
        """
        self.logger.info("Starting CPM Regression Analysis")
        self.logger.info("=" * 50)
        self.logger.info(f"Results Directory:       {self.results_directory}")
        self.logger.info(f"Outer CV strategy:       {self.cv}")
        self.logger.info(f"Inner CV strategy:       {self.inner_cv}")
        self.logger.info(f"Edge selection method:   {self.edge_selection}")
        self.logger.info(f"Select stable edges:     {'Yes' if self.select_stable_edges else 'No'}")
        if self.select_stable_edges:
            self.logger.info(f"Stability threshold:     {self.stability_threshold}")
        self.logger.info(f"Impute Missing Values:   {'Yes' if self.impute_missing_values else 'No'}")
        self.logger.info(f"Number of Permutations:  {self.n_permutations}")
        self.logger.info("=" * 50)

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

        generate_data_insights(X=X, y=y, covariates=covariates, results_directory=self.results_directory)

        X, y, covariates = check_data(X, y, covariates, impute_missings=self.impute_missing_values, device='cpu')

        self.logger.info("=" * 50)

        # Estimate models on permuted data
        p = self.n_permutations + 1  # n_permutations + unpermuted = nperms + 1
        cov_tensor = covariates
        cov_tot = cov_tensor.unsqueeze(0).repeat(p, 1, 1)  # (p, n_samples, n_cov)

        y_tot = torch.zeros((p, y.shape[0]), dtype=y.dtype)
        X_tot = torch.zeros((p, *X.shape), dtype=X.dtype)

        y_tot[0] = y
        X_tot[0] = X

        print("Permuting data..")
        for i in range(1, p):
            idx = torch.randperm(y.shape[0])
            y_tot[i] = y[idx]
            X_tot[i] = X[idx]
            print(i)

        print("Permuted data..")
        torch.cuda.empty_cache()
        gc.collect()

        print("Estimating..")
        import time
        start = time.time()
        results = self._batch_run(X=X_tot, y=y_tot, covariates=cov_tot)
        end = time.time()
        self.logger.info(f"Estimation took {end - start:.2f} seconds")

        results_managers = []
        for b in range(len(results["metrics_detailed"])):
            rm = ResultsManager(output_dir=self.results_directory, perm_run=b,
                                n_folds=self.cv.get_n_splits(), n_features=X.shape[-1])
            results_managers.append(rm)

        for b, results_manager in enumerate(results_managers):
            for fold in range(self.cv.get_n_splits()):
                edge_mask = results["edge_masks"][b][fold].cpu().numpy()

                edges = {
                    'positive': np.where(edge_mask > 0)[0],
                    'negative': np.array([], dtype=int)
                }

                fold_key = f'fold_{fold}'
                scores = results["metrics_detailed"][b][fold_key]

                results_manager.store_edges(edges=edges, fold=fold)
                results_manager.store_metrics(metrics=results["metrics_detailed"][b][f'fold_{fold}'], params={},
                                              fold=fold, param_id=0)

                y_pred = results["predictions"][b][fold_key]
                y_true = y_tot[b][fold * (y.shape[0] // self.cv.get_n_splits()):(fold + 1) * (
                        y.shape[0] // self.cv.get_n_splits())].cpu().numpy()
                test_idx = np.arange(y_true.shape[0])
                results_manager.store_predictions(
                    y_pred=y_pred,
                    y_true=y_true,
                    params={},
                    fold=fold,
                    param_id=0,
                    test_indices=test_idx
                )

                network_strengths = results["network_strengths"][b][fold_key]
                y_true = y_tot[b][fold * (y.shape[0] // self.cv.get_n_splits()):
                                  (fold + 1) * (y.shape[0] // self.cv.get_n_splits())].cpu().numpy()

                results_manager.store_network_strengths(
                    network_strengths=network_strengths,
                    y_true=y_true,
                    fold=fold
                )

            results_manager.calculate_final_cv_results()
            results_manager.calculate_edge_stability()

            results_manager.save_predictions()
            results_manager.save_network_strengths()

            self.logger.info(results_manager.agg_results.round(4).to_string())

        # reporter = HTMLReporter(results_directory=self.results_directory, atlas_labels=self.atlas_labels)
        # reporter.generate_html_report()

        if self.n_permutations > 0:
            PermutationManager.calculate_permutation_results(self.results_directory, self.logger)
        self.logger.info("Estimation completed.")
        self.logger.info("Generating results file.")
        reporter = HTMLReporter(results_directory=self.results_directory, atlas_labels=self.atlas_labels)
        reporter.generate_html_report()

        endend = time.time()
        print(f"Total time: {(endend - start):.2f} seconds")

    def _get_safe_batch_size(self, X: torch.Tensor, safety_factor: float = 0.6) -> int:
        """
        Estimate a safe batch size based on available GPU/CPU memory.
        Uses safety_factor (0.6 means use 60% of available memory).
        """
        element_size = X.element_size()  # bytes per element
        total_elements = X[0].numel()  # elements per permutation
        bytes_per_batch = total_elements * element_size

        if torch.cuda.is_available():
            mem_info = torch.cuda.mem_get_info()
            free_mem = mem_info[0]  # bytes
        else:
            free_mem = psutil.virtual_memory().available

        max_batches = max(int((free_mem * safety_factor) // bytes_per_batch), 1)
        return max_batches

    def _batch_run(self,
                   X: torch.Tensor,
                   y: torch.Tensor,
                   covariates: torch.Tensor) -> dict:
        """
        Run CPM with batched edge statistics across permutations and folds.
        X: [B, n, p]
        y: [B, n]
        covariates: [B, n, c]
        """
        import time
        import gc

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        B, n, p = X.shape
        _, _, c = covariates.shape
        n_folds = self.cv.get_n_splits()

        batch_size = self._get_safe_batch_size(X)
        batch_size = 1
        self.logger.info(f"Auto-detected safe batch size: {batch_size} (B={B})")

        dataset = TensorDataset(X, y, covariates)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        edge_masks_all = []
        metrics_all = []
        all_predictions_dict = []
        all_network_strengths = []

        # Edge statistic function
        def edge_stat_fn(Xf, yf, covf):
            r, p_val = EdgeStatistic.edge_statistic_fn(
                Xf, yf, covf,
                edge_statistic="pearson",
                t_test_filter=self.edge_selection.t_test_filter
            )
            return r, p_val

        start_total = time.time()

        for batch_idx, (X_batch, y_batch, cov_batch) in enumerate(loader):
            self.logger.info(f"Processing batch {batch_idx + 1}/{len(loader)}")

            #X_batch = X_batch
            #y_batch = y_batch
            #cov_batch = cov_batch  # .to(device)
            batch_B = X_batch.shape[0]

            # Loop over permutations in this batch
            for b in range(batch_B):
                self.logger.info(f"  Permutation {b + 1 + batch_idx * batch_size}/{B}")
                Xb, yb, covb = X_batch[b], y_batch[b], cov_batch[b]
                # Xb = Xb.to(device)
                # yb = yb.to(device)
                # covb = covb.to(device)

                # Collect all train/test splits for folds
                folds = list(self.cv.split(torch.arange(n)))
                n_folds = len(folds)

                Xtr_all = torch.stack([Xb[train_idx] for train_idx, _ in folds], dim=0)  # [n_folds, n_train, p] .to(device)
                ytr_all = torch.stack([yb[train_idx] for train_idx, _ in folds], dim=0)
                covtr_all = torch.stack([covb[train_idx] for train_idx, _ in folds], dim=0)

                Xte_all = torch.stack([Xb[test_idx] for _, test_idx in folds], dim=0)
                yte_all = torch.stack([yb[test_idx] for _, test_idx in folds], dim=0)
                covte_all = torch.stack([covb[test_idx] for _, test_idx in folds], dim=0)

                gc.collect()

                def vmap_edge_stat_in_chunks(Xtr_all, ytr_all, covtr_all, chunk_size=4):
                    """
                    Apply vmap in chunks to avoid OOM.
                    Xtr_all, ytr_all, covtr_all: [n_folds, n_train, p] / [n_folds, n_train] / [n_folds, n_train, c]
                    chunk_size: number of folds to process at once
                    """
                    n_folds = Xtr_all.shape[0]
                    r_edges_list = []
                    p_edges_list = []

                    import time
                    tik = time.time()

                    for start in range(0, n_folds, chunk_size):
                        end = min(start + chunk_size, n_folds)
                        self.logger.info("    Calculating folds {start}-{end}".format(start=start, end=end))
                        X_chunk = Xtr_all[start:end].to(device)
                        y_chunk = ytr_all[start:end].to(device)
                        cov_chunk = covtr_all[start:end].to(device)

                        r_chunk, p_chunk = torch.vmap(edge_stat_fn)(X_chunk, y_chunk, cov_chunk)
                        r_edges_list.append(r_chunk)
                        p_edges_list.append(p_chunk)

                        # free GPU memory
                        del X_chunk, y_chunk, cov_chunk
                        torch.cuda.empty_cache()
                        gc.collect()

                    tok = time.time()
                    self.logger.info(f"    Took {tok - tik:.2f} seconds")

                    r_edges = torch.cat(r_edges_list, dim=0)
                    p_edges = torch.cat(p_edges_list, dim=0)
                    return r_edges, p_edges

                def estimate_vram_per_fold(n_train, p, c, dtype_bytes=4):
                    # Xtr + ytr + covtr + outputs (roughly)
                    return (n_train * (p + 1 + c) + p) * dtype_bytes * 1 # * 0.5 is risky but speeds up

                def get_free_gpu_memory(device=None):
                    if device is None:
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    if device.type != 'cuda':
                        return float('inf')
                    free_bytes, _ = torch.cuda.mem_get_info()
                    return free_bytes

                def compute_safe_chunk_size(n_train, p, c, device=None, dtype_bytes=4, safety_factor=0.85):
                    """Compute maximum number of folds to process at once"""
                    free_bytes = get_free_gpu_memory(device)
                    vram_per_fold = estimate_vram_per_fold(n_train, p, c, dtype_bytes)

                    self.logger.info(f"    Estimated VRAM per fold: {vram_per_fold / 1e6:.2f} MB")
                    self.logger.info(f"    Free GPU memory: {free_bytes / 1e6:.2f} MB")
                    chunk_size = max(1, int(free_bytes * safety_factor / vram_per_fold))
                    return chunk_size

                chunk_size = compute_safe_chunk_size(n_train=Xtr_all.shape[1],
                                                     p=Xtr_all.shape[2],
                                                     c=covtr_all.shape[2],
                                                     device=device)
                self.logger.info(f"    -> Safe to use chunksize: {chunk_size} chunks")

                #chunk_size = 1
                r_edges, p_edges = vmap_edge_stat_in_chunks(Xtr_all, ytr_all, covtr_all, chunk_size=chunk_size)

                # Compute edge statistics over folds with vmap
                #r_edges, p_edges = torch.vmap(edge_stat_fn)(Xtr_all, ytr_all, covtr_all)  # [n_folds, p]

                # Loop over folds for model fitting and evaluation
                metrics_fold = {}
                predictions_fold = {}
                strengths_fold = {}
                edge_masks_fold = torch.zeros(n_folds, p, dtype=torch.bool)

                for fold_idx in range(n_folds):
                    Xtr, ytr, covtr = Xtr_all[fold_idx].to(device), ytr_all[fold_idx].to(device), covtr_all[fold_idx].to(device)
                    Xte, yte, covte = Xte_all[fold_idx].to(device), yte_all[fold_idx].to(device), covte_all[fold_idx].to(device)

                    edge_mask = (p_edges[fold_idx] < 0.01).to(dtype=torch.bool)
                    edge_masks_fold[fold_idx] = edge_mask

                    model = LinearCPMModel(device=device).fit(Xtr, ytr, covtr, edge_mask=edge_mask)
                    y_pred_dict = model.predict(Xte, covte, edge_mask=edge_mask)

                    scores = score_regression_models(
                        y_true=yte,
                        y_pred_dict=y_pred_dict,
                        primary_metric_only=False
                    )
                    strengths = model.get_network_strengths(Xte, covte, edge_mask=edge_mask)

                    metrics_fold[f'fold_{fold_idx}'] = scores
                    predictions_fold[f'fold_{fold_idx}'] = y_pred_dict
                    strengths_fold[f'fold_{fold_idx}'] = strengths

                    del Xtr, ytr, covtr, Xte, yte, covte, model
                    torch.cuda.empty_cache()
                    gc.collect()

                edge_masks_all.append(edge_masks_fold.detach().cpu())
                metrics_all.append(metrics_fold)
                all_predictions_dict.append(predictions_fold)
                all_network_strengths.append(strengths_fold)

                del Xtr_all, ytr_all, covtr_all
                del Xte_all, yte_all, covte_all
                torch.cuda.empty_cache()
                gc.collect()

            del X_batch, y_batch, cov_batch
            torch.cuda.empty_cache()
            gc.collect()

        end_total = time.time()
        self.logger.info(f"Total _batch_run time: {(end_total - start_total):.2f} seconds")

        return {
            "edge_masks": edge_masks_all,  # list of [n_folds, p] tensors
            "metrics_detailed": metrics_all,
            "predictions": all_predictions_dict,
            "network_strengths": all_network_strengths,
        }


