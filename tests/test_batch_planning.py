import torch

from cccpm.batch_planning import plan_batch_sizes, make_cpm_cost_fn, estimate_bytes, BatchPlan
from cccpm.results_manager import ResultsManager


def test_estimate_bytes_monotonic_in_each_dimension():
    """Cost must never decrease when any single batch dimension grows --
    plan_batch_sizes' binary search relies on this."""
    cost_fn = make_cpm_cost_fn(n_samples_train=100, n_samples_test=25, n_features=500, n_cov=3)

    base = cost_fn(1, 1, 1)
    assert cost_fn(2, 1, 1) >= base
    assert cost_fn(1, 2, 1) >= base
    assert cost_fn(1, 1, 2) >= base
    assert cost_fn(4, 3, 2) >= cost_fn(2, 2, 2) >= cost_fn(1, 1, 1)


def test_estimate_bytes_includes_results_manager_slice():
    """The results_manager slice term should exactly match
    ResultsManager.estimate_slice_bytes, since batch_planning reuses it."""
    n_features, b_params, b_folds, b_perms = 200, 2, 3, 4
    rm_bytes = ResultsManager.estimate_slice_bytes(n_features, b_params, b_folds, b_perms)
    total = estimate_bytes(n_samples_train=50, n_samples_test=10, n_features=n_features, n_cov=2,
                           b_params=b_params, b_folds=b_folds, b_perms=b_perms, overhead_multiplier=1.0)
    assert total >= rm_bytes


def test_plan_batch_sizes_tiny_budget_gives_all_ones():
    """When even the (1,1,1) baseline barely fits, nothing should be able to grow."""
    cost_fn = make_cpm_cost_fn(n_samples_train=200, n_samples_test=50, n_features=2000, n_cov=3)
    baseline_cost = cost_fn(1, 1, 1)
    plan = plan_batch_sizes(n_params=10, n_folds=10, n_perms=100, cost_fn=cost_fn,
                            available_bytes=baseline_cost / 0.8)
    assert plan == BatchPlan(params=1, folds=1, perms=1)


def test_plan_batch_sizes_huge_budget_gives_full_batch():
    cost_fn = make_cpm_cost_fn(n_samples_train=50, n_samples_test=10, n_features=100, n_cov=2)
    plan = plan_batch_sizes(n_params=5, n_folds=10, n_perms=20, cost_fn=cost_fn,
                            available_bytes=10**14)
    assert plan == BatchPlan(params=5, folds=10, perms=20)


def test_plan_batch_sizes_escalates_params_before_folds_before_perms():
    """With a budget that fits full params but not full params+folds, only
    params should grow; folds/perms should stay at 1."""
    cost_fn = make_cpm_cost_fn(n_samples_train=50, n_samples_test=10, n_features=100, n_cov=2)
    n_params, n_folds, n_perms = 5, 10, 20

    budget_params_only = cost_fn(n_params, 1, 1) / 0.8
    plan = plan_batch_sizes(n_params, n_folds, n_perms, cost_fn, available_bytes=budget_params_only)
    assert plan.params == n_params
    assert plan.folds == 1

    # A larger budget that fits full params+folds but not full perms should
    # grow folds next, without needing to touch perms.
    budget_params_and_folds = cost_fn(n_params, n_folds, 1) / 0.8
    plan2 = plan_batch_sizes(n_params, n_folds, n_perms, cost_fn, available_bytes=budget_params_and_folds)
    assert plan2.params == n_params
    assert plan2.folds == n_folds


def test_plan_batch_sizes_never_exceeds_totals():
    cost_fn = make_cpm_cost_fn(n_samples_train=30, n_samples_test=10, n_features=50, n_cov=1)
    plan = plan_batch_sizes(n_params=3, n_folds=4, n_perms=5, cost_fn=cost_fn, available_bytes=10**20)
    assert plan.params <= 3
    assert plan.folds <= 4
    assert plan.perms <= 5


def test_available_memory_bytes_cpu_positive():
    from cccpm.batch_planning import available_memory_bytes
    assert available_memory_bytes('cpu') > 0
