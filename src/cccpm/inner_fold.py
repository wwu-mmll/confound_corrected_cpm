import torch

from cccpm.utils import torch_train_test_split, build_fold_batch
from cccpm.scoring import score_models, score_models_batched
from cccpm.results_manager import ResultsManager
from cccpm.edge_selection import BaseEdgeSelector
from cccpm.batch_planning import plan_batch_sizes, make_cpm_cost_fn, available_memory_bytes, BatchPlan


def _group_params_by_correction(param_grid):
    """
    Group param_grid entries by (edge_selection instance, correction method),
    preserving original param_grid order across and within groups.

    PThreshold.select_batch batches many threshold values in one call, but
    only within a single selector + correction method (different correction
    methods use different formulas). Returns a list of dicts:
    {'selector', 'correction', 'indices': [param_grid index, ...],
     'thresholds': [threshold, ...]} -- 'indices' preserves the original
    param_grid position of each threshold, so results are always written
    back to the correct results_manager slot.
    """
    groups = {}
    order = []
    for idx, config in enumerate(param_grid):
        selector = config['edge_selection']
        correction = config.get('edge_selection__correction')
        key = (id(selector), correction)
        if key not in groups:
            groups[key] = {'selector': selector, 'correction': correction, 'indices': [], 'thresholds': []}
            order.append(key)
        groups[key]['indices'].append(idx)
        groups[key]['thresholds'].append(config['edge_selection__threshold'])
    return [groups[k] for k in order]


def run_inner_folds(cpm_model, X, y, covariates, inner_cv, edge_selection: BaseEdgeSelector,
                    results_directory, device, task_type):

    param_grid = edge_selection.param_grid
    n_features = X.shape[1]
    n_params = len(param_grid)
    n_folds = inner_cv.get_n_splits()
    n_perms = y.shape[1]

    with torch.cuda.nvtx.range("inner_cv:results_manager_init"):
        results_manager = ResultsManager(
            output_dir=results_directory, n_runs=n_perms,
            n_folds=n_folds, n_features=n_features, n_params=n_params,
            device=device)

    with torch.cuda.nvtx.range("inner_cv:data_to_gpu"):
        X_gpu = torch.as_tensor(X, device=device, dtype=torch.float32)
        y_gpu = torch.as_tensor(y, device=device, dtype=torch.float32)
        cov_gpu = torch.as_tensor(covariates, device=device, dtype=torch.float32)

    # Non-linear models (sklearn/pygam-backed) cannot participate in torch
    # batching at all -- they always use the exact original per-item loop.
    supports_batching = hasattr(cpm_model, 'fit_batched') and hasattr(cpm_model, 'predict_batched')

    if not supports_batching:
        for fold_id, (train, test) in enumerate(inner_cv.split(X, y[:, 0])):

            with torch.cuda.nvtx.range(f"inner_cv:fold{fold_id}:split"):
                X_train, X_test, y_train, y_test, cov_train, cov_test = torch_train_test_split(
                    train, test, X_gpu, y_gpu, cov_gpu)

            for param_id, config in enumerate(param_grid):
                with torch.cuda.nvtx.range(f"inner_cv:fold{fold_id}:param{param_id}:threshold"):
                    edge_selection.set_params(**config)
                    selected_edges = edge_selection.fit_transform(
                        X_train, y_train, cov_train, device=device).return_selected_edges()

                with torch.cuda.nvtx.range(f"inner_cv:fold{fold_id}:param{param_id}:model_fit"):
                    y_pred = (cpm_model(edges=selected_edges, device=device, task_type=task_type)
                              .fit(X_train, y_train, cov_train)
                              .predict(X_test, cov_test, return_proba=True))

                with torch.cuda.nvtx.range(f"inner_cv:fold{fold_id}:param{param_id}:scoring"):
                    metrics = score_models(y_true=y_test, y_pred=y_pred,
                                           task_type=task_type, device=device)

                with torch.cuda.nvtx.range(f"inner_cv:fold{fold_id}:param{param_id}:store"):
                    results_manager.store_edges(param_idx=param_id, fold_idx=fold_id,
                                                edges_tensor=selected_edges)
                    results_manager.store_metrics(param_idx=param_id, fold_idx=fold_id,
                                                  metrics_tensor=metrics)

    else:
        splits = list(inner_cv.split(X, y[:, 0]))
        n_cov = cov_gpu.shape[1]
        n_samples_train = max(len(tr) for tr, te in splits)
        n_samples_test = max(len(te) for tr, te in splits)

        cost_fn = make_cpm_cost_fn(n_samples_train, n_samples_test, n_features, n_cov)
        avail = available_memory_bytes(device)
        with torch.cuda.nvtx.range("inner_cv:plan_batch_sizes"):
            plan = plan_batch_sizes(n_params, n_folds, n_perms, cost_fn, avail)

        param_groups = _group_params_by_correction(param_grid)

        for fold_start in range(0, n_folds, plan.folds):
            fold_ids = list(range(fold_start, min(fold_start + plan.folds, n_folds)))
            fold_splits = [splits[i] for i in fold_ids]

            with torch.cuda.nvtx.range(f"inner_cv:foldbatch{fold_start}:split"):
                fb = build_fold_batch(X_gpu, y_gpu, cov_gpu, fold_splits)

            for perm_start in range(0, n_perms, plan.perms):
                perm_end = min(perm_start + plan.perms, n_perms)
                perm_slice = slice(perm_start, perm_end)
                y_train_p = fb.y_train[:, :, perm_slice]
                y_test_p = fb.y_test[:, :, perm_slice]

                # r/p don't depend on params -- computed once per (fold-batch, perm-batch)
                # and reused across every param group/batch below.
                with torch.cuda.nvtx.range(
                        f"inner_cv:foldbatch{fold_start}:permbatch{perm_start}:fit_transform"):
                    r_edges, p_edges = edge_selection.edge_statistic.fit_transform_batched(
                        X=fb.X_train, y=y_train_p, covariates=fb.cov_train,
                        valid_mask=fb.train_valid, device=device)

                for group in param_groups:
                    selector = group['selector']
                    n_group_params = len(group['indices'])
                    for pg_start in range(0, n_group_params, max(plan.params, 1)):
                        pg_end = min(pg_start + max(plan.params, 1), n_group_params)
                        idx_batch = group['indices'][pg_start:pg_end]
                        thresholds = group['thresholds'][pg_start:pg_end]

                        with torch.cuda.nvtx.range(
                                f"inner_cv:foldbatch{fold_start}:permbatch{perm_start}:"
                                f"paramgroup{pg_start}:threshold"):
                            selector.correction = group['correction']
                            selected_edges = selector.select_batch(r=r_edges, p=p_edges, thresholds=thresholds)

                        with torch.cuda.nvtx.range(
                                f"inner_cv:foldbatch{fold_start}:permbatch{perm_start}:"
                                f"paramgroup{pg_start}:model_fit"):
                            model = cpm_model(edges=selected_edges, device=device, task_type=task_type)
                            model.fit_batched(fb.X_train, y_train_p, fb.cov_train, valid_mask=fb.train_valid)
                            y_pred = model.predict_batched(fb.X_test, fb.cov_test,
                                                            valid_mask=fb.test_valid, return_proba=True)

                        with torch.cuda.nvtx.range(
                                f"inner_cv:foldbatch{fold_start}:permbatch{perm_start}:"
                                f"paramgroup{pg_start}:scoring"):
                            metrics = score_models_batched(y_true=y_test_p, y_pred=y_pred, task_type=task_type,
                                                            valid_mask=fb.test_valid, device=device)

                        with torch.cuda.nvtx.range(
                                f"inner_cv:foldbatch{fold_start}:permbatch{perm_start}:"
                                f"paramgroup{pg_start}:store"):
                            # idx_batch is contiguous by construction: ParameterGrid
                            # enumerates one selector's sub-grid at a time, and within a
                            # selector, correction (sorted before threshold) is the
                            # outer-varying key, so a (selector, correction) group's
                            # param_grid indices are always a contiguous run. Assert it
                            # explicitly -- writing to the wrong slot would be silent
                            # data corruption, not a crash.
                            assert idx_batch == list(range(idx_batch[0], idx_batch[-1] + 1)), (
                                f"param_grid indices for one (selector, correction) group must be "
                                f"contiguous, got {idx_batch}")
                            param_slice = slice(idx_batch[0], idx_batch[-1] + 1)
                            fold_slice = slice(fold_ids[0], fold_ids[-1] + 1)
                            results_manager.store_edges(param_idx=param_slice, fold_idx=fold_slice,
                                                        edges_tensor=selected_edges, run_idx=perm_slice)
                            results_manager.store_metrics(param_idx=param_slice, fold_idx=fold_slice,
                                                          metrics_tensor=metrics, run_idx=perm_slice)

    with torch.cuda.nvtx.range("inner_cv:aggregate"):
        results_manager.aggregate_inner_folds()

    with torch.cuda.nvtx.range("inner_cv:find_best"):
        best_param_id = results_manager.find_best_params(task_type=task_type)

    with torch.cuda.nvtx.range("inner_cv:build_best_params"):
        best_params = [param_grid[i] for i in best_param_id.tolist()]

    with torch.cuda.nvtx.range("inner_cv:stability"):
        stability_edges = results_manager.calculate_edge_stability(
            write=False, best_param_id=best_param_id)

    return best_params, stability_edges
