"""Inner cross-validation loop for edge-selection hyperparameter tuning."""
import torch

from cccpm.preprocessing import torch_train_test_split
from cccpm.scoring import score_models
from cccpm.results_manager import ResultsManager
from cccpm.edge_selection import (BaseEdgeSelector, filter_connected_components,
                                  resolve_min_component_size)


def _group_params_by_correction(param_grid):
    """Groups param_grid entries by (edge_selection instance, correction), since `PThreshold.select` batches thresholds only within one such group.

    Returns a list of dicts {'selector', 'correction', 'indices': [param_grid
    index, ...], 'thresholds': [...]}; 'indices' maps results back to their slot.
    """
    groups = {}
    order = []
    for idx, config in enumerate(param_grid):
        selector = config['edge_selection']
        correction = config.get('edge_selection__correction')
        key = (id(selector), correction)
        if key not in groups:
            groups[key] = {'selector': selector, 'correction': correction,
                           'indices': [], 'thresholds': []}
            order.append(key)
        groups[key]['indices'].append(idx)
        groups[key]['thresholds'].append(config['edge_selection__threshold'])
    return [groups[k] for k in order]


def run_inner_folds(cpm_model, X, y, covariates, inner_cv, edge_selection: BaseEdgeSelector,
                    results_directory, device, task_type):
    """Runs inner CV over the edge-selection hyperparameter grid; returns (winning config per run, its per-edge selection stability)."""
    param_grid = edge_selection.param_grid
    n_features = X.shape[1]
    n_params = len(param_grid)
    n_folds = inner_cv.get_n_splits()
    n_perms = y.shape[1]

    # Only the fold-averaged stability of the best param is needed, so keep
    # just the fold-sum (no per-fold edges.npy) -- this runs once per outer fold.
    results_manager = ResultsManager(
        output_dir=results_directory, n_runs=n_perms, n_folds=n_folds,
        n_features=n_features, n_params=n_params, device=device,
        store_fold_edges=False)

    X_dev = torch.as_tensor(X, device=device, dtype=torch.float32)
    y_dev = torch.as_tensor(y, device=device, dtype=torch.float32)
    cov_dev = torch.as_tensor(covariates, device=device, dtype=torch.float32)

    param_groups = _group_params_by_correction(param_grid)
    min_edges = resolve_min_component_size(edge_selection.connected_components)

    for fold_id, (train, test) in enumerate(inner_cv.split(X, y[:, 0])):
        X_train, X_test, y_train, y_test, cov_train, cov_test = torch_train_test_split(
            train, test, X_dev, y_dev, cov_dev)

        # Statistics don't depend on the threshold, so computed once per fold.
        r_edges, t_edges, df = edge_selection.statistic.fit_transform(
            X=X_train, y=y_train, covariates=cov_train, device=device)

        for group in param_groups:
            selector = group['selector']
            selector.correction = group['correction']
            selected_edges = selector.select(r=r_edges, t=t_edges, df=df,
                                             thresholds=group['thresholds'])
            if min_edges is not None:
                selected_edges = filter_connected_components(selected_edges, min_edges)

            # Thresholding is batched per group, but the model fit is not:
            # batching was measured to be worth nothing here
            # (scripts/benchmark_batching.py).
            for local_idx, param_idx in enumerate(group['indices']):
                edges_p = selected_edges[:, :, local_idx, :]

                model = cpm_model(edges=edges_p, device=device, task_type=task_type)
                y_pred = (model.fit(X_train, y_train, cov_train)
                               .predict(X_test, cov_test, return_proba=True))
                metrics = score_models(y_true=y_test, y_pred=y_pred,
                                       task_type=task_type, device=device)

                results_manager.store_edges(param_idx=param_idx, fold_idx=fold_id,
                                            edges_tensor=edges_p)
                results_manager.store_metrics(param_idx=param_idx, fold_idx=fold_id,
                                              metrics_tensor=metrics)

    results_manager.aggregate_inner_folds()
    best_param_id = results_manager.find_best_params(task_type=task_type)
    best_params = [param_grid[i] for i in best_param_id.tolist()]
    stability_edges = results_manager.calculate_edge_stability(
        write=False, best_param_id=best_param_id)

    return best_params, stability_edges
