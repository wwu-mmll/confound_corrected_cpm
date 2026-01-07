import functorch

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
            metrics = score_regression_models(y_true=y_test, y_pred_dict=y_pred)

            results_manager.store_edges(selected_edges, fold_id, param_id)
            results_manager.store_metrics(metrics=metrics, params=config, fold=fold_id, param_id=param_id)

    # once all outer folds are done, calculate final results and edge stability
    results_manager.aggregate_inner_folds()

    best_params, best_param_id = results_manager.find_best_params()
    stability_edges = results_manager.calculate_edge_stability(write=False, best_param_id=best_param_id)

    return best_params, stability_edges


def run_inner_folds_torch(X, y, covariates, inner_cv, edge_selection: BaseEdgeSelector, results_directory, perm_run):
    """
    Run inner cross-validation over all folds and hyperparameter configurations.

    Parameters
    ----------
    X : torch.Tensor
        Input features, shape [n_samples, n_features].
    y : torch.Tensor
        Target values, shape [n_samples].
    covariates : torch.Tensor
        Covariate matrix, shape [n_samples, n_covariates].
    inner_cv : CV splitter
        Inner cross-validation object with get_n_splits() and split().
    edge_selection : BaseEdgeSelector
        Edge selection object with param_grid and fit_transform().
    results_directory : str
        Directory to save results.
    perm_run : int
        Permutation run index (for bookkeeping).

    Returns
    -------
    best_params : dict
        Best hyperparameter configuration according to inner CV.
    stability_edges : dict
        Dictionary with 'positive' and 'negative' keys mapping to edge stability scores.
    """

    param_grid = edge_selection.param_grid
    n_features = X.shape[1]
    n_params = len(param_grid)
    n_folds = inner_cv.get_n_splits()

    results_manager = ResultsManager(
        output_dir=results_directory,
        perm_run=perm_run,
        n_folds=n_folds,
        n_features=n_features,
        n_params=n_params
    )

    # Loop over inner folds
    for fold_id, (train_idx, test_idx) in enumerate(inner_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        cov_train, cov_test = covariates[train_idx], covariates[test_idx]

        # Loop over hyperparameter configs
        for param_id, config in enumerate(param_grid):
            edge_selection.set_params(**config)

            # Fit edge selection and get selected edges
            r_masked, p_masked, valid_edges = edge_selection.fit_transform(X_train, y_train, cov_train)
            selected_edges = edge_selection.return_selected_edges(r=r_masked, p=p_masked)

            # Fit CPM model on selected edges
            model = LinearCPMModel()
            model.fit(
                X_train, y_train, cov_train,
                pos_edges=selected_edges["positive"],
                neg_edges=selected_edges["negative"]
            )

            # Predict on test set
            y_pred = model.predict(
                X_test, cov_test,
                pos_edges=selected_edges["positive"],
                neg_edges=selected_edges["negative"]
            )

            # Compute metrics
            metrics = score_regression_models(y_true=y_test, y_pred_dict=y_pred)

            # Save edges and metrics
            results_manager.store_edges(selected_edges, fold=fold_id, param_id=param_id)
            results_manager.store_metrics(metrics=metrics, params=config, fold=fold_id, param_id=param_id)

    # Aggregate results across folds
    results_manager.aggregate_inner_folds()

    # Find best hyperparameters and edge stability
    best_params, best_param_id = results_manager.find_best_params()
    stability_edges = results_manager.calculate_edge_stability(write=False, best_param_id=best_param_id)

    return best_params, stability_edges
