import torch

from cccpm.utils import torch_train_test_split
from cccpm.scoring import score_models
from cccpm.results_manager import ResultsManager
from cccpm.edge_selection import (BaseEdgeSelector, filter_connected_components,
                                  resolve_min_component_size)


def _group_params_by_correction(param_grid):
    """
    Group param_grid entries by (edge_selection instance, correction method),
    preserving original param_grid order across and within groups.

    `PThreshold.select` evaluates many threshold values in one call, but only
    within a single selector + correction method (different corrections use
    different formulas). Returns a list of dicts:
    {'selector', 'correction', 'indices': [param_grid index, ...],
     'thresholds': [threshold, ...]} -- 'indices' preserves each threshold's
    original param_grid position, so results are written back to the right slot.
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
    """
    Inner cross-validation over the edge-selection hyperparameter grid.

    Returns the winning configuration per run, plus the per-edge selection
    stability of that configuration (used for stable-edge selection).
    """
    param_grid = edge_selection.param_grid
    n_features = X.shape[1]
    n_params = len(param_grid)
    n_folds = inner_cv.get_n_splits()
    n_perms = y.shape[1]

    # Inner CV only needs the fold-averaged stability of the best param, so keep
    # just the fold-sum (no per-fold edges.npy) -- cheaper for many
    # params/folds/perms, and this runs once per outer fold.
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

        # r/p don't depend on the selection threshold, so they are computed once
        # per fold and reused across every parameter group below.
        r_edges, p_edges = edge_selection.edge_statistic.fit_transform(
            X=X_train, y=y_train, covariates=cov_train, device=device)

        for group in param_groups:
            selector = group['selector']
            selector.correction = group['correction']
            selected_edges = selector.select(r=r_edges, p=p_edges,
                                             thresholds=group['thresholds'])
            if min_edges is not None:
                selected_edges = filter_connected_components(selected_edges, min_edges)

            # selected_edges is [Features, 2, N_thresholds, Runs]; the model
            # fits one configuration at a time. Thresholding is batched (the
            # multiple-comparison correction is applied once for the whole
            # group); the model fit is not, because it is a handful of OLS
            # solves on a design matrix ~3 columns wide -- batching it was
            # measured to be worth nothing on the much larger folds axis
            # (scripts/benchmark_batching.py), and this axis is smaller still.
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
