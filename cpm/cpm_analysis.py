import os
import logging
import shutil
import gc

from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import psutil
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit, KFold

from cpm.logging import setup_logging
from cpm.models import LinearCPMModel
from cpm.edge_selection import UnivariateEdgeSelection, PThreshold, EdgeStatistic, BaseEdgeSelector
from cpm.results_manager import ResultsManager, PermutationManager
from cpm.utils import check_data, impute_missing_values, select_stable_edges, generate_data_insights
from cpm.scoring import score_regression_models
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
            #  idx = torch.randperm(y.shape[0])
            np.random.seed(i)
            idx_np = np.random.permutation(y.shape[0])
            idx = torch.from_numpy(idx_np).to(y.device)
            y_tot[i] = y[idx]
            X_tot[i] = X[idx]
            #print(y[idx])
            #print(X[idx])

        #print(y_tot)
        #print(y_tot)

        self.logger.info("Permuted data..")
        torch.cuda.empty_cache()
        #gc.collect()

        self.logger.info("Estimating..")
        import time
        start = time.time()
        results = self._batch_run(X=X_tot, y=y_tot, covariates=cov_tot, edge_selection=self.edge_selection)
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

                results_manager.store_edges(edges=edges, fold=fold)
                results_manager.store_metrics(metrics=results["metrics_detailed"][b][f'fold_{fold}'], params={},
                                              fold=fold, param_id=0)

                y_pred = results["predictions"][b][fold_key]
                test_idx = results["test_indices"][b][fold_key]
                y_true = y_tot[b][test_idx].cpu().numpy()
                results_manager.store_predictions(
                    y_pred=y_pred,
                    y_true=y_true,
                    params={},
                    fold=fold,
                    param_id=0,
                    test_indices=test_idx
                )

                network_strengths = results["network_strengths"][b][fold_key]
                y_true = y_tot[b].cpu().numpy()

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

    # X_tot: [p, n, f]
    # Contains feature data (X) for all permutations.
    # p = number of permutations + 1 (includes original data)
    # n = number of samples
    # f = number of features

    # y_tot: [p, n]
    # Target variable (y) for all permutations.
    # First row is the original data, remaining rows are permuted versions.

    # cov_tot: [p, n, c]
    # Covariate data for all permutations.
    # c = number of covariates

    # X_batch, y_batch, cov_batch: [B, n, f], [B, n], [B, n, c]
    # A batch of B permutations selected from X_tot, y_tot, cov_tot for processing.

    # Xtr_all: [B, n_folds, n_train, f]
    # Training feature data for each permutation and fold.
    # n_folds = number of cross-validation folds
    # n_train = number of training samples per fold

    # ytr_all: [B, n_folds, n_train]
    # Training target values corresponding to Xtr_all.

    # covtr_all: [B, n_folds, n_train, c]
    # Training covariates corresponding to Xtr_all.

    # Xte_all: [B, n_folds, n_test, f]
    # Test feature data for each permutation and fold.

    # yte_all: [B, n_folds, n_test]
    # Test target values corresponding to Xte_all.

    # covte_all: [B, n_folds, n_test, c]
    # Test covariates corresponding to Xte_all.

    # edge_masks_fold: [n_folds, f]
    # Binary mask indicating selected edges (features) per fold.
    # True where p-value < threshold (e.g., 0.01)

    # predictions_fold: Dict[str, np.ndarray]
    # Predicted values for each fold, e.g., {'fold_0': y_pred_array}

    # metrics_fold: Dict[str, Dict]
    # Evaluation metrics for each fold, e.g., R², MAE, etc.

    # strengths_fold: Dict[str, np.ndarray]
    # Network strengths computed from selected edges per fold.

    # test_indices_fold: Dict[str, np.ndarray]
    # Indices of test samples per fold, used to align predictions with ground truth.
    def _batch_run(self,
                   X: torch.Tensor,
                   y: torch.Tensor,
                   covariates: torch.Tensor,
                   edge_selection: BaseEdgeSelector) -> dict:
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
        n_outer_folds = n_folds

        batch_size = self._get_safe_batch_size(X)
        self.logger.info(f"Auto-detected safe batch size: {batch_size} (B={B})")

        dataset = TensorDataset(X, y, covariates)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Containers for final results across all permutations
        edge_masks_all = []
        metrics_all = []
        all_predictions_dict = []
        all_network_strengths = []
        all_test_indices = []

        def edge_stat_fn(Xf, yf, covf):
            r, p_val = EdgeStatistic.edge_statistic_fn(
                Xf, yf, covf,
                edge_statistic=self.edge_selection.edge_statistic.edge_statistic,
                t_test_filter=self.edge_selection.t_test_filter
            )
            return r, p_val

        def estimate_vram_per_fold(n_train, p, c, dtype_bytes=4):
            return (n_train * (p + 1 + c) + p) * dtype_bytes

        def get_free_gpu_memory():
            if torch.cuda.is_available():
                free_bytes, _ = torch.cuda.mem_get_info()
            else:
                free_bytes = psutil.virtual_memory().available
            return free_bytes

        def compute_safe_chunk_size(n_train, p, c, dtype_bytes=4, safety_factor=0.7):
            """Compute maximum number of folds to process at once"""
            free_bytes = get_free_gpu_memory()
            vram_per_fold = estimate_vram_per_fold(n_train, p, c, dtype_bytes)

            self.logger.info(f"    Estimated VRAM per fold: {vram_per_fold / 1e6:.2f} MB")
            self.logger.info(f"    Free GPU memory: {free_bytes / 1e6:.2f} MB")
            chunk_size = max(1, int(free_bytes * safety_factor / vram_per_fold))
            return chunk_size

        def compute_safe_perm_batch_size(n_folds, n_train, p, c, dtype_bytes=4, safety_factor=0.7):
            free_bytes = get_free_gpu_memory()
            vram_per_perm = n_folds * estimate_vram_per_fold(n_train, p, c, dtype_bytes)  # VRAM per permutation
            self.logger.info(f"    Estimated VRAM per permutation: {vram_per_perm / 1e6:.2f} MB")
            self.logger.info(f"    Free GPU memory: {free_bytes / 1e6:.2f} MB")
            batch_size = max(1, int(free_bytes * safety_factor / vram_per_perm))
            return batch_size

        def vmap_edge_stat_in_chunks(Xtr_all, ytr_all, covtr_all, chunk_size=4):
            B, n_folds, _, _ = Xtr_all.shape
            r_edges_list = []
            p_edges_list = []

            tik = time.time()
            for start in range(0, n_folds, chunk_size):
                end = min(start + chunk_size, n_folds)
                X_chunk = Xtr_all[:, start:end].to(device)
                y_chunk = ytr_all[:, start:end].to(device)
                cov_chunk = covtr_all[:, start:end].to(device)

                torch.cuda.empty_cache()
                self.logger.info(f"    Calculating folds {start}-{end} on device {device}")

                r_chunk, p_chunk = torch.vmap(torch.vmap(edge_stat_fn))(X_chunk, y_chunk, cov_chunk)

                r_chunk = r_chunk.cpu()
                p_chunk = p_chunk.cpu()

                r_edges_list.append(r_chunk)
                p_edges_list.append(p_chunk)

                del X_chunk, y_chunk, cov_chunk, r_chunk, p_chunk
                torch.cuda.empty_cache()
                #gc.collect()

            tok = time.time()
            self.logger.info(f"    Took {tok - tik:.2f} seconds")

            r_edges = torch.cat(r_edges_list, dim=1)
            p_edges = torch.cat(p_edges_list, dim=1)
            return r_edges, p_edges

        start_total = time.time()

        for batch_idx, (X_batch, y_batch, cov_batch) in enumerate(loader):
            self.logger.info(f"Processing batch {batch_idx + 1}/{len(loader)}")
            batch_B = X_batch.shape[0]

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
                B_sub = Xb_batch.shape[0]

                folds = list(self.cv.split(torch.arange(n)))
                n_folds = len(folds)

                # print("folds:")
                # print(n_folds)
                # print(folds)

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

                #print(Xtr_all[0, 0][:5], ytr_all[0, 0][:5])

                # Compute r/p edge statistics for each perm in the sub-batch and each outer fold
                safe = .7
                success = False
                while not success:  # this is a memory-critical computation
                    try:
                        self.logger.debug(f"Trying chunk factor {safe}")
                        chunk_size = compute_safe_chunk_size(n, p, c, safety_factor=safe)
                        r_edges, p_edges = vmap_edge_stat_in_chunks(Xtr_all, ytr_all, covtr_all, chunk_size=chunk_size)
                        # print("edges:")
                        # print(r_edges[0, 0], p_edges[0, 0])
                        #
                        # alpha = 0.05
                        #
                        # r0 = r_edges[0, 0, :]
                        # p0 = p_edges[0, 0, :]
                        #
                        # pos = torch.where((r0 > 0) & (p0 < alpha))[0].cpu().numpy()
                        # neg = torch.where((r0 < 0) & (p0 < alpha))[0].cpu().numpy()
                        #
                        # print("Positive Edges:", pos)
                        # print("Negative Edges:", neg)
                    except Exception as exc:
                        self.logger.warning(f"Edge stat computation failed at factor {safe}: {exc}")
                        if not safe <= .1:
                            safe -= .1
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise Exception("Failed to compute edge stats")
                    else:
                        success = True

                # For each permutation in this sub-batch, run inner-CV on the OUTER-TRAIN set,
                # pick best hyperparameters, then evaluate the outer fold using those best params.

                for batch_idx, (X_batch, y_batch, cov_batch) in enumerate(loader):
                    self.logger.info(f"Processing batch {batch_idx + 1}/{len(loader)}")
                    batch_B = X_batch.shape[0]

                    perm_batch_size = compute_safe_perm_batch_size(n_outer_folds, n_train=n, p=p, c=c)
                    self.logger.info(f"    -> Safe to use permutation batch size: {perm_batch_size}")

                    for start_idx in range(0, batch_B, perm_batch_size):
                        end_idx = min(start_idx + perm_batch_size, batch_B)
                        self.logger.info(
                            f"  Processing permutations {start_idx + 1 + batch_idx * batch_size}-{end_idx + batch_idx * batch_size}")

                        # slice current permutation batch (torch tensors)
                        Xb_batch = X_batch[start_idx:end_idx]  # shape [B_sub, n, p]
                        yb_batch = y_batch[start_idx:end_idx]  # shape [B_sub, n]
                        covb_batch = cov_batch[start_idx:end_idx]  # shape [B_sub, n, c]
                        B_sub = Xb_batch.shape[0]

                        # prepare outer-fold train/test splits (list of (train_idx, test_idx) wrt sample axis 0..n-1)
                        folds = list(self.cv.split(torch.arange(n)))
                        n_folds_local = len(folds)

                        # Build per-permutation, per-outer-fold train/test tensors:
                        # Xtr_all: [B_sub, n_folds_local, n_train_fold, p]
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

                        # Compute r/p edge statistics for each perm in the sub-batch and each outer fold
                        safe = 0.7
                        success = False
                        while not success:
                            try:
                                self.logger.debug(f"Trying chunk factor {safe}")
                                chunk_size = compute_safe_chunk_size(n, p, c, safety_factor=safe)
                                r_edges, p_edges = vmap_edge_stat_in_chunks(Xtr_all, ytr_all, covtr_all,
                                                                            chunk_size=chunk_size)
                                # r_edges, p_edges shapes: [B_sub, n_folds_local, p]
                            except Exception as exc:
                                self.logger.warning(f"Edge stat computation failed at factor {safe}: {exc}")
                                if safe > 0.1:
                                    safe -= 0.1
                                    torch.cuda.empty_cache()
                                    continue
                                else:
                                    raise
                            else:
                                success = True

                        # For each permutation in this sub-batch, run inner-CV on the OUTER-TRAIN set,
                        # pick best hyperparameters, then evaluate the outer fold using those best params.
                        for perm_local_idx in range(B_sub):
                            global_perm_idx = batch_idx * batch_size + start_idx + perm_local_idx

                            # Create a ResultsManager that will store outer-fold results for this permutation
                            results_manager = ResultsManager(
                                output_dir=self.results_directory,
                                perm_run=global_perm_idx,
                                n_folds=n_folds_local,
                                n_features=p,
                                n_params=len(edge_selection.param_grid)
                            )

                            # PERMUTATION-LEVEL containers
                            perm_edge_masks = torch.zeros(n_folds_local, p, dtype=torch.bool)  # final outer-fold masks
                            perm_metrics = {}
                            perm_predictions = {}
                            perm_strengths = {}
                            perm_test_indices = {}

                            # For each outer fold, run inner CV on the OUTER TRAINING SET
                            for outer_fold_idx, (train_idx_outer, test_idx_outer) in enumerate(folds):
                                # Extract outer-train and outer-test tensors (torch)
                                X_train_outer = Xtr_all[perm_local_idx, outer_fold_idx]  # shape [n_train_fold, p]
                                y_train_outer = ytr_all[perm_local_idx, outer_fold_idx]  # shape [n_train_fold]
                                cov_train_outer = covtr_all[perm_local_idx, outer_fold_idx]  # [n_train_fold, c]

                                X_test_outer = Xte_all[perm_local_idx, outer_fold_idx]
                                y_test_outer = yte_all[perm_local_idx, outer_fold_idx]
                                cov_test_outer = covte_all[perm_local_idx, outer_fold_idx]

                                # record test indices (as numpy) for alignment later
                                perm_test_indices[f'fold_{outer_fold_idx}'] = (
                                    folds[outer_fold_idx][1].cpu().numpy()
                                    if isinstance(folds[outer_fold_idx][1], torch.Tensor)
                                    else np.array(folds[outer_fold_idx][1])
                                )

                                # Convert outer-train to numpy for edge_selection (many selectors expect numpy)
                                X_train_outer_np = X_train_outer.cpu().numpy()
                                y_train_outer_np = y_train_outer.cpu().numpy()
                                cov_train_outer_np = cov_train_outer.cpu().numpy() if cov_train_outer is not None else None

                                # INNER CV hyperparameter search on this outer-train
                                if self.inner_cv:
                                    param_grid = edge_selection.param_grid
                                    n_params = len(param_grid)
                                    inner_n_splits = self.inner_cv.get_n_splits()

                                    # results manager for inner CV (temporary)
                                    inner_rm = ResultsManager(
                                        output_dir=os.path.join(results_manager.results_directory, 'inner_cv',
                                                                str(outer_fold_idx)),
                                        perm_run=global_perm_idx,
                                        n_folds=inner_n_splits,
                                        n_features=p,
                                        n_params=n_params
                                    )

                                    # Build inner-fold splits relative to the outer-train sample indices:
                                    n_train_outer = X_train_outer_np.shape[0]
                                    inner_indices = np.arange(n_train_outer)

                                    for inner_fold_id, (inner_train_idx, inner_test_idx) in enumerate(
                                            self.inner_cv.split(inner_indices)):
                                        # inner train/test data (numpy)
                                        X_inner_tr_np = X_train_outer_np[inner_train_idx]
                                        X_inner_te_np = X_train_outer_np[inner_test_idx]
                                        y_inner_tr_np = y_train_outer_np[inner_train_idx]
                                        y_inner_te_np = y_train_outer_np[inner_test_idx]
                                        cov_inner_tr_np = cov_train_outer_np[
                                            inner_train_idx] if cov_train_outer_np is not None else None
                                        cov_inner_te_np = cov_train_outer_np[
                                            inner_test_idx] if cov_train_outer_np is not None else None

                                        # For each hyperparameter config: run selector -> fit CPM -> evaluate
                                        for param_id, config in enumerate(param_grid):
                                            edge_selection.set_params(**config)

                                            # fit selector on inner-train and get selected edges
                                            X_inner_tr_b = X_inner_tr_np[None, ...]  # shape (1, n_inner, p)
                                            y_inner_tr_b = y_inner_tr_np[None, ...]  # shape (1, n_inner)
                                            cov_inner_tr_b = (
                                                cov_inner_tr_np[None, ...] if cov_inner_tr_np is not None else None)

                                            sel_out = edge_selection.fit_transform(X_inner_tr_b, y_inner_tr_b,
                                                                                   cov_inner_tr_b)
                                            selected_edges = sel_out.return_selected_edges()

                                            # Normalize returned format: allow either boolean mask (p,) or indices
                                            # selected_edges may be (1, p) boolean too — squeeze to (p,)
                                            if isinstance(selected_edges, np.ndarray) and selected_edges.ndim == 2 and \
                                                    selected_edges.shape[0] == 1:
                                                selected_edges = selected_edges.squeeze(0)

                                            # convert selected_edges (indices or boolean) to torch boolean mask on device
                                            mask_tensor = torch.zeros(p, dtype=torch.bool, device=device)
                                            if isinstance(selected_edges,
                                                          np.ndarray) and selected_edges.dtype == bool and \
                                                    selected_edges.shape[0] == p:
                                                mask_tensor = torch.from_numpy(selected_edges).to(device)
                                            else:
                                                if selected_edges is not None and len(selected_edges) > 0:
                                                    #print(selected_edges)
                                                    all_edges = torch.cat([selected_edges['positive'], selected_edges['negative']])
                                                    mask_tensor[all_edges] = True
                                                    #mask_tensor[selected_edges] = True

                                            # Prepare torch inner-train/test for CPM fit/predict (on device)
                                            X_inner_tr_t = torch.tensor(X_inner_tr_np, device=device)
                                            y_inner_tr_t = torch.tensor(y_inner_tr_np, device=device)
                                            cov_inner_tr_t = torch.tensor(cov_inner_tr_np,
                                                                          device=device) if cov_inner_tr_np is not None else None
                                            X_inner_te_t = torch.tensor(X_inner_te_np, device=device)
                                            y_inner_te_t = torch.tensor(y_inner_te_np, device=device)
                                            cov_inner_te_t = torch.tensor(cov_inner_te_np,
                                                                          device=device) if cov_inner_te_np is not None else None

                                            # Fit CPM on inner-train using mask_tensor and evaluate on inner-test
                                            model = LinearCPMModel(device=device).fit(X_inner_tr_t, y_inner_tr_t,
                                                                                      cov_inner_tr_t,
                                                                                      edge_mask=mask_tensor)
                                            y_pred_inner = model.predict(X_inner_te_t, cov_inner_te_t,
                                                                         edge_mask=mask_tensor)

                                            # compute metrics (use numpy arrays for ResultsManager consistency)
                                            #print(y_inner_te_np, y_pred_inner)
                                            metrics_inner = score_regression_models(
                                                y_true=y_inner_te_np,
                                                y_pred_dict=y_pred_inner
                                            )

                                            # store in inner ResultsManager
                                            inner_rm.store_edges(selected_edges, inner_fold_id, param_id)
                                            inner_rm.store_metrics(metrics=metrics_inner, params=config,
                                                                   fold=inner_fold_id, param_id=param_id)

                                            # cleanup
                                            del model, X_inner_tr_t, y_inner_tr_t, X_inner_te_t, y_inner_te_t
                                            torch.cuda.empty_cache()

                                    # finished inner folds for this outer fold -> aggregate & choose best
                                    inner_rm.aggregate_inner_folds()
                                    best_params, best_param_id = inner_rm.find_best_params()
                                    stability_edges = inner_rm.calculate_edge_stability(write=False,
                                                                                        best_param_id=best_param_id)
                                else:
                                    # no inner CV configured -> use first param config as default
                                    best_params = edge_selection.param_grid[0]
                                    stability_edges = None

                                # Now with best_params (or stability), compute final selected edges on the FULL outer-train
                                if self.select_stable_edges and (stability_edges is not None):
                                    # select_stable_edges should accept result of stability_edges
                                    edges_selected_final = select_stable_edges(stability_edges,
                                                                               self.stability_threshold)
                                    # edges_selected_final expected to be an index array or boolean mask
                                else:
                                    edge_selection.set_params(**best_params)
                                    # --- FIX A: add batch dim for the full outer-train selection ---
                                    X_train_outer_b = X_train_outer_np[None, ...]  # (1, n_train_outer, p)
                                    y_train_outer_b = y_train_outer_np[None, ...]  # (1, n_train_outer)
                                    cov_train_outer_b = (
                                        cov_train_outer_np[None, ...] if cov_train_outer_np is not None else None)

                                    sel_out_full = edge_selection.fit_transform(X_train_outer_b, y_train_outer_b,
                                                                                cov_train_outer_b)
                                    edges_selected_final = sel_out_full.return_selected_edges()

                                    # if returned shape is (1, p) squeeze to (p,)
                                    if isinstance(edges_selected_final, np.ndarray) and getattr(edges_selected_final,
                                                                                                "ndim", None) == 2 and \
                                            edges_selected_final.shape[0] == 1:
                                        edges_selected_final = edges_selected_final.squeeze(0)

                                # convert edges_selected_final to torch mask
                                final_mask = torch.zeros(p, dtype=torch.bool, device=device)

                                if isinstance(edges_selected_final,
                                              np.ndarray) and edges_selected_final.dtype == bool and \
                                        edges_selected_final.shape[0] == p:
                                    final_mask = torch.from_numpy(edges_selected_final).to(device)
                                elif edges_selected_final is not None:
                                    # Handle 'positive' and 'negative' separately
                                    for key in ['positive', 'negative']:
                                        if key in edges_selected_final and len(edges_selected_final[key]) > 0:
                                            idx = edges_selected_final[key]
                                            if isinstance(idx, torch.Tensor):
                                                idx = idx.to(device).long()
                                            else:
                                                idx = torch.tensor(idx, device=device, dtype=torch.long)
                                            final_mask[idx] = True

                                # store the mask for this outer fold (on CPU)
                                perm_edge_masks[outer_fold_idx] = final_mask.cpu()

                                # Fit final CPM on outer-train and evaluate on outer-test
                                model_final = LinearCPMModel(device=device).fit(
                                    X_train_outer.to(device), y_train_outer.to(device),
                                    cov_train_outer.to(device) if cov_train_outer is not None else None,
                                    edge_mask=final_mask
                                )
                                y_pred_outer = model_final.predict(X_test_outer.to(device), cov_test_outer.to(
                                    device) if cov_test_outer is not None else None, edge_mask=final_mask)
                                strengths_outer = model_final.get_network_strengths(X_test_outer.to(device),
                                                                                    cov_test_outer.to(
                                                                                        device) if cov_test_outer is not None else None,
                                                                                    edge_mask=final_mask)

                                # compute metrics (prefer numpy for consistency)
                                # convert y_test_outer and y_pred_outer to numpy if needed
                                if isinstance(y_pred_outer, dict):
                                    metrics_outer = score_regression_models(y_true=y_test_outer.cpu().numpy(),
                                                                            y_pred_dict=y_pred_outer,
                                                                            primary_metric_only=False)
                                else:
                                    metrics_outer = score_regression_models(y_true=y_test_outer.cpu().numpy(),
                                                                            y_pred=y_pred_outer)

                                # Store fold-level results in the permutation ResultsManager
                                edges_dict = {
                                    'positive': edges_selected_final['positive'],
                                    'negative': edges_selected_final['negative']
                                }
                                results_manager.store_edges(edges=edges_dict, fold=outer_fold_idx)
                                results_manager.store_predictions(
                                    y_pred=y_pred_outer,
                                    y_true=y_test_outer.cpu().numpy(),
                                    params=best_params,
                                    fold=outer_fold_idx,
                                    param_id=0,
                                    test_indices=perm_test_indices[f'fold_{outer_fold_idx}']
                                )
                                results_manager.store_metrics(metrics=metrics_outer, params=best_params,
                                                              fold=outer_fold_idx, param_id=0)
                                results_manager.store_network_strengths(network_strengths=strengths_outer,
                                                                        y_true=y_test_outer.cpu().numpy(),
                                                                        fold=outer_fold_idx)

                                # cleanup per outer fold
                                del model_final
                                torch.cuda.empty_cache()

                            # End loop over outer folds for this permutation
                            # finalize permutation-level ResultsManager
                            results_manager.calculate_final_cv_results()
                            results_manager.calculate_edge_stability()

                            # collect results for return
                            edge_masks_all.append(perm_edge_masks.detach().cpu())
                            metrics_all.append(results_manager.agg_results)
                            all_predictions_dict.append(results_manager.cv_predictions)
                            all_network_strengths.append(results_manager.cv_network_strengths)
                            #all_test_indices.append(results_manager.test_indices)

                            # log
                            self.logger.info(
                                f"Permutation {global_perm_idx}: best params summary: (see inner rm where available)")
                            try:
                                self.logger.info(results_manager.agg_results.round(4).to_string())
                            except Exception:
                                pass

                            # cleanup
                            del results_manager
                            torch.cuda.empty_cache()

                        # freed per-sub-batch tensors
                        del Xtr_all, ytr_all, covtr_all, Xte_all, yte_all, covte_all
                        torch.cuda.empty_cache()

                    # freed this loader batch
                    #del X_batch, y_batch, cov_batch
                    torch.cuda.empty_cache()

                # for perm_idx in range(end - start):
                #     metrics_fold = {}
                #     predictions_fold = {}
                #     strengths_fold = {}
                #     edge_masks_fold = torch.zeros(n_folds, p, dtype=torch.bool)
                #     test_indices_fold = {}
                #
                #     for fold_idx in range(n_folds):
                #         _, test_idx = folds[fold_idx]
                #         test_indices_fold[f'fold_{fold_idx}'] = test_idx.cpu().numpy() if (
                #             isinstance(test_idx, torch.Tensor)) else np.array(test_idx)
                #         Xtr = Xtr_all[perm_idx, fold_idx].to(device)
                #         ytr = ytr_all[perm_idx, fold_idx].to(device)
                #         covtr = covtr_all[perm_idx, fold_idx].to(device)
                #
                #         Xte = Xte_all[perm_idx, fold_idx].to(device)
                #         yte = yte_all[perm_idx, fold_idx].to(device)
                #         covte = covte_all[perm_idx, fold_idx].to(device)
                #
                #         edge_mask = (p_edges[perm_idx, fold_idx] < 0.01).to(dtype=torch.bool)
                #         edge_masks_fold[fold_idx] = edge_mask
                #         # Fit and evaluate CPM model
                #         model = LinearCPMModel(device=device).fit(Xtr, ytr, covtr, edge_mask=edge_mask)
                #         y_pred_dict = model.predict(Xte, covte, edge_mask=edge_mask)
                #
                #         scores = score_regression_models(
                #             y_true=yte,
                #             y_pred_dict=y_pred_dict,
                #             primary_metric_only=False
                #         )
                #         strengths = model.get_network_strengths(Xte, covte, edge_mask=edge_mask)
                #         # Store fold results
                #         metrics_fold[f'fold_{fold_idx}'] = scores
                #         predictions_fold[f'fold_{fold_idx}'] = y_pred_dict
                #         strengths_fold[f'fold_{fold_idx}'] = strengths
                #
                #         del Xtr, ytr, covtr, Xte, yte, covte, model
                #         torch.cuda.empty_cache()
                #         #gc.collect()
                #
                #     # Store permutation-level results
                #     edge_masks_all.append(edge_masks_fold.detach().cpu())
                #     metrics_all.append(metrics_fold)
                #     all_predictions_dict.append(predictions_fold)
                #     all_network_strengths.append(strengths_fold)
                #     all_test_indices.append(test_indices_fold)
                #
                # del Xtr_all, ytr_all, covtr_all, Xte_all, yte_all, covte_all
                # torch.cuda.empty_cache()
                # #gc.collect()

            #del X_batch, y_batch, cov_batch
            torch.cuda.empty_cache()
            #gc.collect()

        end_total = time.time()
        self.logger.info(f"Total _batch_run time: {(end_total - start_total):.2f} seconds")

        return {
            "edge_masks": edge_masks_all,
            "metrics_detailed": metrics_all,
            "predictions": all_predictions_dict,
            "network_strengths": all_network_strengths,
            "test_indices": all_test_indices
        }
