from cpm.models import LinearCPMModel
from cpm.utils import train_test_split
from cpm.scoring import score_regression_models
from cpm.results_manager import ResultsManager
from cpm.edge_selection import BaseEdgeSelector, EdgeStatistic

import torch


def run_inner_folds(X, y, covariates, inner_cv, edge_selection: BaseEdgeSelector, results_directory,
                    perm_run):
    """
    Run inner cross-validation over all folds and hyperparameter configurations.

    Returns
    -------
    cv_results : DataFrame
        Aggregated results from all inner folds.
    stability_edges : dict
        Dictionary with 'positive' and 'negative' keys mapping to arrays of edge stability scores.
    """
    param_grid = edge_selection.param_grid
    n_features = X.shape[1]
    n_params = len(param_grid)
    n_folds = inner_cv.get_n_splits()

    results_manager = ResultsManager(output_dir=results_directory, perm_run=perm_run,
                                     n_folds=n_folds, n_features=n_features, n_params=n_params)

    for fold_id, (train, test) in enumerate(inner_cv.split(X, y)):
        # split according to single fold
        X_train, X_test, y_train, y_test, cov_train, cov_test = train_test_split(train, test, X, y, covariates)

        for param_id, config in enumerate(param_grid):
            edge_selection.set_params(**config)
            selected_edges = edge_selection.fit_transform(X_train, y_train, cov_train).return_selected_edges()
            y_pred = LinearCPMModel(edges=selected_edges).fit(X_train, y_train, cov_train).predict(X_test, cov_test)
            metrics = score_regression_models(y_true=y_test, y_pred=y_pred)

            results_manager.store_edges(selected_edges, fold_id, param_id)
            results_manager.store_metrics(metrics=metrics, params=config, fold=fold_id, param_id=param_id)

    # once all outer folds are done, calculate final results and edge stability
    results_manager.aggregate_inner_folds()

    best_params, best_param_id = results_manager.find_best_params()
    stability_edges = results_manager.calculate_edge_stability(write=False, best_param_id=best_param_id)

    return best_params, stability_edges


def run_inner_folds_cuda(X, y, covariates, inner_cv, edge_selection: BaseEdgeSelector, results_directory,
                    perms):


    return

def run_inner_folds_gpu(X, y, covariates,
                        Xte=None, yte=None, covte=None,
                        inner_cv=None, edge_selection=None,
                        results_directory=None, perm_run=None, device=None):
    """
    GPU-compatible version of run_inner_folds.

    X: (B, F, n_train, p)
    y: (B, F, n_train)
    covariates: (B, F, n_train, c)

    Xte: (B, F, n_test, p)
    yte: (B, F, n_test)
    covte: (B, F, n_test, c)
    """
    device = torch.device(device if device is not None else "cuda")

    B, n_folds, n_train, p = X.shape
    c = covariates.shape[-1]

    param_grid = edge_selection.param_grid
    n_params = len(param_grid)
    n_features = p

    # Results manager per permutation
    results_managers = [
        ResultsManager(
            output_dir=results_directory,
            perm_run=perm_run + b,
            n_folds=n_folds,
            n_features=n_features,
            n_params=n_params,
        )
        for b in range(B)
    ]

    for fold_id in range(n_folds):
        X_train = X[:, fold_id].to(device)
        y_train = y[:, fold_id].to(device)
        cov_train = covariates[:, fold_id].to(device)

        X_test = Xte[:, fold_id].to(device) if Xte is not None else None
        y_test = yte[:, fold_id].to(device) if yte is not None else None
        cov_test = covte[:, fold_id].to(device) if covte is not None else None

        for param_id, config in enumerate(param_grid):
            edge_selection.set_params(**config)

            r_edges, p_edges = EdgeStatistic.edge_statistic_fn(
                X_train, y_train, cov_train,
                edge_statistic=edge_selection.edge_statistic.edge_statistic,
                t_test_filter=edge_selection.t_test_filter
            )

            alpha = getattr(edge_selection.edge_statistic, "alpha", 0.05)
            edge_mask = (p_edges < alpha)

            for b in range(B):
                selected_edges = edge_mask[b].detach().cpu().numpy()

                results_managers[b].store_edges(
                    selected_edges=selected_edges,
                    fold=fold_id,
                    param_id=param_id
                )

                if X_test is not None and y_test is not None and cov_test is not None:
                    model = LinearCPMModel(device=device).fit(
                        X_train[b], y_train[b], cov_train[b], edge_mask=edge_mask[b]
                    )

                    y_pred = model.predict(
                        X_test[b], cov_test[b], edge_mask=edge_mask[b]
                    )

                    metrics = score_regression_models(
                        y_true=y_test[b], y_pred_dict=y_pred
                    )

                    results_managers[b].store_metrics(
                        metrics=metrics,
                        params=config,
                        fold=fold_id,
                        param_id=param_id
                    )

    best_params_list = []
    stability_edges_list = []

    for b in range(B):
        rm = results_managers[b]
        rm.aggregate_inner_folds()
        best_params, best_param_id = rm.find_best_params()
        stability_edges = rm.calculate_edge_stability(write=False, best_param_id=best_param_id)
        best_params_list.append(best_params)
        stability_edges_list.append(stability_edges)

    return best_params_list, stability_edges_list


