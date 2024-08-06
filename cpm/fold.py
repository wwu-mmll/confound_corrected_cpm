import pandas as pd

from cpm.models import LinearCPMModel, NetworkDict, ModelDict
from cpm.utils import score_regression_models, train_test_split


def compute_inner_fold(X, y, covariates, cv, edge_selection, param, param_id):
    cv_results = pd.DataFrame()
    edge_selection.set_params(**param)
    for fold_id, (nested_train, nested_test) in enumerate(cv.split(X, y)):
        (X_train, X_test, y_train,
         y_test, cov_train, cov_test) = train_test_split(nested_train, nested_test, X, y, covariates)

        res = compute_fold(X_train, X_test, y_train, y_test, cov_train, cov_test, edge_selection, param, param_id, fold_id)
        cv_results = pd.concat([cv_results, pd.DataFrame(res)], ignore_index=True)

    cv_results.set_index(['fold', 'param_id', 'network', 'model'], inplace=True)
    cv_results.sort_index(inplace=True)
    return cv_results


def compute_fold(X_train, X_test, y_train, y_test, cov_train, cov_test, edge_selection, param, param_id, fold_id):
    edges = edge_selection.fit_transform(X=X_train, y=y_train, covariates=cov_train)
    model = LinearCPMModel(edges=edges).fit(X_train, y_train, cov_train)
    y_pred = model.predict(X_test, cov_test)
    metrics = score_regression_models(y_true=y_test, y_pred=y_pred)
    cv_results = pd.DataFrame()
    for model_type in ModelDict().keys():
        for network in NetworkDict().keys():
            res = metrics[model_type][network]
            res['model'] = model_type
            res['network'] = network
            res['fold'] = fold_id
            res['param_id'] = param_id
            res['params'] = [param]
            cv_results = pd.concat([cv_results, pd.DataFrame(res, index=[0])], ignore_index=True)
    return cv_results
