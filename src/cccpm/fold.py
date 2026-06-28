from cccpm.utils import train_test_split
from cccpm.scoring import score_regression_models
from cccpm.results_manager import ResultsManager
from cccpm.edge_selection import BaseEdgeSelector


def run_inner_folds(cpm_model, X, y, covariates, inner_cv, edge_selection: BaseEdgeSelector, results_directory,
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
            y_pred = cpm_model(edges=selected_edges).fit(X_train, y_train, cov_train).predict(X_test, cov_test)
            metrics = score_regression_models(y_true=y_test, y_pred=y_pred)

            results_manager.store_edges(selected_edges, fold_id, param_id)
            results_manager.store_metrics(metrics=metrics, params=config, fold=fold_id, param_id=param_id)

    # once all outer folds are done, calculate final results and edge stability
    results_manager.aggregate_inner_folds()

    best_params, best_param_id = results_manager.find_best_params()
    stability_edges = results_manager.calculate_edge_stability(write=False, best_param_id=best_param_id)

    return best_params, stability_edges
