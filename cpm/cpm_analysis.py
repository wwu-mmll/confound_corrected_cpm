import os
import logging
import shutil

from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import psutil
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit, KFold

#from cpm import edge_selection
from cpm.logging import setup_logging
from cpm.models import LinearCPMModel
from cpm.edge_selection import UnivariateEdgeSelection, PThreshold, EdgeStatistic, BaseEdgeSelector
from cpm.results_manager import ResultsManager, PermutationManager
from cpm.utils import check_data, impute_missing_values, select_stable_edges, generate_data_insights
from cpm.scoring import score_regression_models
from cpm.fold import run_inner_folds_torch
from cpm.reporting import HTMLReporter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
                     edge_selection=PThreshold(threshold=[0.05], correction=None)
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

        np.random.seed(42)
        torch.manual_seed(42)
        os.makedirs(self.results_directory, exist_ok=True)
        os.makedirs(os.path.join(self.results_directory, "edges"), exist_ok=True)
        setup_logging(os.path.join(self.results_directory, "cpm_log.txt"))
        self.logger = logging.getLogger(__name__)

        self._log_analysis_details()

        if self.inner_cv is None:
            if len(self.edge_selection.param_grid) > 1:
                raise RuntimeError("Multiple hyperparameter configurations but no inner cv defined. "
                                   "Please provide only one hyperparameter configuration or an inner cv.")
            if self.select_stable_edges:
                raise RuntimeError("Stable edges can only be selected when using an inner cv.")

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

        if not os.path.isfile(csv_path):
            raise RuntimeError(f"CSV file does not exist: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
            missing = required_columns - set(df.columns)

            if missing:
                raise RuntimeError(f"CSV file is missing required columns: {', '.join(missing)}")
        except Exception as e:
            raise RuntimeError(f"Error reading CSV file {csv_path}: {e}")

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

        p = self.n_permutations + 1  # n_permutations + unpermuted = nperms + 1
        cov_tensor = covariates
        cov_tot = cov_tensor.unsqueeze(0).repeat(p, 1, 1)  # (p, n_samples, n_cov)

        y_tot = torch.zeros((p, y.shape[0]), dtype=y.dtype)
        X_tot = torch.zeros((p, *X.shape), dtype=X.dtype)

        y_tot[0] = y
        X_tot[0] = X

        self.logger.info("Permuting data..")
        for i in range(1, p):
            np.random.seed(i)
            idx_np = np.random.permutation(y.shape[0])
            idx = torch.from_numpy(idx_np).to(y.device)
            y_tot[i] = y[idx]
            X_tot[i] = X[idx]

        self.logger.info("Permuted data..")
        torch.cuda.empty_cache()

        self.logger.info("Estimating..")
        import time
        start = time.time()
        self._run(X=X_tot, y=y_tot, covariates=cov_tot)
        end = time.time()
        self.logger.info(f"Estimation took {end - start:.2f} seconds")

        if self.n_permutations > 0:
            PermutationManager.calculate_permutation_results(self.results_directory, self.logger)
        self.logger.info("Estimation completed.")
        self.logger.info("Generating results file.")
        reporter = HTMLReporter(results_directory=self.results_directory, atlas_labels=self.atlas_labels)
        reporter.generate_html_report()

        endend = time.time()
        self.logger.info("Total runtime: {:.2f} seconds".format(endend - start))

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

    def _run(self,
                   X: torch.Tensor,
                   y: torch.Tensor,
                   covariates: torch.Tensor):
        """
        Run CPM with batched edge statistics across permutations and folds.
        X: [B, n, p]
        y: [B, n]
        covariates: [B, n, c]
        """
        B, n, p = X.shape
        _, _, c = covariates.shape
        n_folds = self.cv.get_n_splits()

        batch_size = self._get_safe_batch_size(X)
        self.logger.info(f"Auto-detected safe batch size: {batch_size} (B={B})")

        dataset = TensorDataset(X, y, covariates)  # split dataset into subsets of smaller size
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        def estimate_vram_per_fold(n_train, p, c, dtype_bytes=4):
            return (n_train * (p + 1 + c) + p) * dtype_bytes

        def get_free_gpu_memory():
            if torch.cuda.is_available():
                free_bytes, _ = torch.cuda.mem_get_info()
            else:
                free_bytes = psutil.virtual_memory().available
            return free_bytes

        def compute_safe_perm_batch_size(n_folds, n_train, p, c, dtype_bytes=4, safety_factor=0.7):
            free_bytes = get_free_gpu_memory()
            vram_per_perm = n_folds * estimate_vram_per_fold(n_train, p, c, dtype_bytes)  # VRAM per permutation
            self.logger.info(f"    Estimated VRAM per permutation: {vram_per_perm / 1e6:.2f} MB")
            self.logger.info(f"    Free GPU memory: {free_bytes / 1e6:.2f} MB")
            batch_size = max(1, int(free_bytes * safety_factor / vram_per_perm))
            return batch_size

        for batch_idx, (X_batch, y_batch, cov_batch) in enumerate(loader):  # do for every permutation subset:
            self.logger.info(f"Processing batch {batch_idx + 1}/{len(loader)}")
            batch_B = X_batch.shape[0]  # no. of permutations in the current subset

            perm_batch_size = compute_safe_perm_batch_size(n_folds, n_train=n, p=p, c=c)
            self.logger.info(f"    -> Safe to use permutation batch size: {perm_batch_size}")

            for start in range(0, batch_B, perm_batch_size):
                end = min(start + perm_batch_size, batch_B)
                self.logger.info(
                    f"  Processing permutations {start + 1 + batch_idx * batch_size}-{end + batch_idx * batch_size}")

                # Slice current permutation batch
                Xb_batch = X_batch[start:end]
                yb_batch = y_batch[start:end]
                covb_batch = cov_batch[start:end]

                folds = list(self.cv.split(torch.arange(n)))
                n_folds = len(folds)

                # Prepare training and test splits for each fold and permutation
                Xtr_all = torch.stack([
                    torch.stack([Xb[train_idx] for train_idx, _ in folds], dim=0)
                    for Xb in Xb_batch
                ], dim=0)
                ytr_all = torch.stack([
                    torch.stack([yb[train_idx] for train_idx, _ in folds], dim=0)
                    for yb in yb_batch
                ], dim=0)
                covtr_all = torch.stack([
                    torch.stack([covb[train_idx] for train_idx, _ in folds], dim=0)
                    for covb in covb_batch
                ], dim=0)
                Xte_all = torch.stack([
                    torch.stack([Xb[test_idx] for _, test_idx in folds], dim=0)
                    for Xb in Xb_batch
                ], dim=0)
                yte_all = torch.stack([
                    torch.stack([yb[test_idx] for _, test_idx in folds], dim=0)
                    for yb in yb_batch
                ], dim=0)
                covte_all = torch.stack([
                    torch.stack([covb[test_idx] for _, test_idx in folds], dim=0)
                    for covb in covb_batch
                ], dim=0)

                rm = []
                for batch_perms in range(0, Xtr_all.shape[0]):  # loop thru all permutations in current slice
                    Xi = Xtr_all[batch_perms]  # slice out current permutation
                    yi = ytr_all[batch_perms]
                    covi = covtr_all[batch_perms]

                    Xi_test = Xte_all[batch_perms]
                    yi_test = yte_all[batch_perms]
                    covi_test = covte_all[batch_perms]

                    rm.append(ResultsManager(output_dir=self.results_directory, perm_run=batch_perms,
                                             n_folds=self.cv.get_n_splits(), n_features=X.shape[2]))

                    for outer_folds in range(n_folds):
                        train, test = folds[outer_folds]
                        Xii = Xi[outer_folds]
                        yii = yi[outer_folds]
                        covii = covi[outer_folds]

                        Xii_test = Xi_test[outer_folds]
                        yii_test = yi_test[outer_folds]
                        covii_test = covi_test[outer_folds]

                        if self.impute_missing_values:
                            Xii, Xii_test, covii, covii_test = impute_missing_values(Xii, Xii_test, covii, covii_test)
                            self.logger.info(f"Imputed missing values")

                        if self.inner_cv:
                            best_params, stability_edges = run_inner_folds_torch(Xii, yii, covii, inner_cv=self.inner_cv, edge_selection=self.edge_selection, results_directory=self.results_directory, perm_run=batch_perms)
                            if batch_perms == 0:
                                self.logger.info(f"Best hyperparameters: {best_params}")
                        else:
                            best_params = self.edge_selection.param_grid[0]

                        if self.select_stable_edges:
                            edges = select_stable_edges(stability_edges, self.stability_threshold)
                        else:
                            self.edge_selection.set_params(**best_params)
                            r_masked, p_masked, valid_edges = self.edge_selection.fit_transform(X=Xii, y=yii,
                                                                      covariates=covii)
                            edges = self.edge_selection.return_selected_edges(r_masked, p_masked)

                        rm[batch_perms].store_edges(edges=edges, fold=outer_folds)

                        model = LinearCPMModel().fit(Xii, yii, covii, pos_edges=edges["positive"], neg_edges=edges["negative"])
                        y_pred = model.predict(Xii_test, covii_test, pos_edges=edges["positive"], neg_edges=edges["negative"])
                        network_strengths = model.get_network_strengths(Xii_test, covii_test, pos_edges=edges["positive"], neg_edges=edges["negative"])
                        metrics = score_regression_models(y_true=yii_test, y_pred_dict=y_pred)
                        rm[batch_perms].store_predictions(y_pred=y_pred, y_true=yii_test, params=best_params,
                                                          fold=outer_folds,
                                                          param_id=0, test_indices=test)
                        rm[batch_perms].store_metrics(metrics=metrics, params=best_params, fold=outer_folds, param_id=0)
                        rm[batch_perms].store_network_strengths(network_strengths=network_strengths, y_true=yii_test,
                                                                fold=outer_folds)

                    rm[batch_perms].calculate_final_cv_results()
                    rm[batch_perms].calculate_edge_stability()

                    if batch_perms == 0:
                        self.logger.info(rm[batch_perms].agg_results.round(4).to_string())
                        rm[batch_perms].save_predictions()
                        rm[batch_perms].save_network_strengths()
