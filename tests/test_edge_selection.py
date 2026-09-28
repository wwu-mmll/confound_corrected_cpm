import numpy as np
import pytest
import torch
from cccpm.edge_selection import (
    resolve_presence_threshold,
    resolve_min_component_size,
    filter_connected_components,
    EdgeStatistic,
    UnivariateEdgeSelection,
    PThreshold,
)
from cccpm.constants import Networks


@pytest.mark.parametrize("selection_input", ["raw", "residualized"])
@pytest.mark.parametrize("statistic", ["pearson", "spearman"])
def test_batched_edge_statistics_match_per_column(statistic, selection_input):
    """Computing (r, t) for all target columns at once must equal computing
    each column separately (up to float32 rounding).

    The pipeline batches edge-statistic computation over all runs/permutations
    in a single call (cpm_analysis._select_edges), relying on the fact that the
    statistic for one target column does not depend on the others. This guards
    that assumption: a regression here would change which edges get selected
    across permutations. The two paths are not bit-identical because a batched
    matrix-matrix product accumulates in a different order than the per-column
    matrix-vector product, but they agree to float32 precision (which does not
    change the downstream |t|>critical edge selection in practice — this is also
    exactly what the inner-CV path has always computed).
    """
    rng = np.random.RandomState(0)
    n_samples, n_features, n_runs = 90, 15, 8
    X = rng.randn(n_samples, n_features).astype(np.float32)
    Y = rng.randn(n_samples, n_runs).astype(np.float32)
    covariates = rng.randn(n_samples, 2).astype(np.float32)

    stat = EdgeStatistic(selection_statistic=statistic,
                         selection_input=selection_input)
    device = torch.device('cpu')

    # Batched: all columns at once.
    r_all, t_all, df_all = stat.fit_transform(X=X, y=Y, covariates=covariates, device=device)

    # Per column, exactly as the old loop did.
    for run_id in range(n_runs):
        r_col, t_col, df_col = stat.fit_transform(
            X=X, y=Y[:, [run_id]], covariates=covariates, device=device)
        assert df_col == df_all
        torch.testing.assert_close(r_all[:, [run_id]], r_col, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(t_all[:, [run_id]], t_col, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("selection_input", ["raw", "residualized"])
@pytest.mark.parametrize("statistic,binary_target", [
    ("pearson", False),
    ("spearman", False),
    ("pearson", True),      # a 0/1 target: the point-biserial correlation
])
def test_edge_selection_recovers_signed_edges(statistic, binary_target,
                                              selection_input):
    """
    End-to-end test of the production edge-selection path
    (UnivariateEdgeSelection -> EdgeStatistic dispatch -> PThreshold.select)
    for every supported edge statistic: a strongly positive edge must be
    selected as positive, a strongly negative edge as negative, and the sign
    must not be confused.
    """
    rng = np.random.RandomState(42)
    n_samples, n_features = 200, 10
    X = rng.randn(n_samples, n_features).astype(np.float32)
    # feature 0 drives y up, feature 1 drives y down
    signal = X[:, 0] * 2.0 - X[:, 1] * 2.0 + rng.randn(n_samples) * 0.3
    if binary_target:
        y = (signal > np.median(signal)).astype(np.float32).reshape(-1, 1)
    else:
        y = signal.astype(np.float32).reshape(-1, 1)
    covariates = rng.randn(n_samples, 1).astype(np.float32)

    sel = UnivariateEdgeSelection(
        selection_statistic=statistic,
        selection_input=selection_input,
        edge_selection=[PThreshold(threshold=[0.01], correction=[None])],
    )
    # Configure as a single selector, exactly as the pipeline does
    sel.set_params(**list(sel.param_grid)[0])

    edges = sel.fit_transform(X=X, y=y, covariates=covariates).return_selected_edges()
    # edges shape: [n_features, 2, n_runs]
    assert bool(edges[0, Networks.positive, 0]), f"{statistic}: feature 0 should be positive"
    assert bool(edges[1, Networks.negative, 0]), f"{statistic}: feature 1 should be negative"
    assert not bool(edges[0, Networks.negative, 0]), f"{statistic}: feature 0 should not be negative"
    assert not bool(edges[1, Networks.positive, 0]), f"{statistic}: feature 1 should not be positive"


# --- Presence filter (sparse/structural-zero edge removal) ---------------------

def test_resolve_presence_threshold():
    """bool/float/None inputs map to the right nonzero-fraction threshold."""
    assert resolve_presence_threshold(False) is None
    assert resolve_presence_threshold(None) is None
    assert resolve_presence_threshold(True) == 0.5
    assert resolve_presence_threshold(0.75) == 0.75
    assert resolve_presence_threshold(0.0) == 0.0
    with pytest.raises(ValueError):
        resolve_presence_threshold(1.5)
    with pytest.raises(ValueError):
        resolve_presence_threshold(-0.1)


def _sparse_presence_data(seed=0):
    """X whose columns are nonzero in known fractions of the 100 subjects.

    Column j is nonzero in exactly ``(j + 1) * 10 %`` of subjects (10%..100%),
    with strong correlation to y on the nonzero rows so that, absent the
    presence filter, every column would be selectable.
    """
    rng = np.random.RandomState(seed)
    n = 100
    y = rng.randn(n).astype(np.float32)
    n_features = 10
    X = np.zeros((n, n_features), dtype=np.float32)
    for j in range(n_features):
        k = (j + 1) * 10  # number of nonzero subjects: 10, 20, ..., 100
        rows = rng.choice(n, size=k, replace=False)
        X[rows, j] = (y[rows] * 2.0 + rng.randn(k) * 0.1).astype(np.float32)
    return X, y.reshape(-1, 1)


@pytest.mark.parametrize("threshold,expected_min_fraction", [
    (0.5, 0.5),
    (0.3, 0.3),
    (0.75, 0.75),
])
def test_presence_filter_drops_sparse_edges(threshold, expected_min_fraction):
    """Only edges nonzero in >= threshold of subjects survive the filter."""
    X, y = _sparse_presence_data()
    stat = EdgeStatistic(selection_statistic='pearson', presence_filter=threshold)
    r, t, _ = stat.fit_transform(X=X, y=y, covariates=None, device=torch.device('cpu'))

    presence = (X != 0).mean(axis=0)
    kept = (t[:, 0].numpy() != 0.0)  # t==0 marks a filtered/invalid edge
    for j in range(X.shape[1]):
        if presence[j] + 1e-9 >= expected_min_fraction:
            assert kept[j], f"edge {j} (presence {presence[j]:.2f}) should survive"
        else:
            assert not kept[j], f"edge {j} (presence {presence[j]:.2f}) should be filtered"


def test_presence_filter_off_by_default_keeps_all():
    """With the filter off, sparse-but-variable edges are still evaluated."""
    X, y = _sparse_presence_data()
    stat = EdgeStatistic(selection_statistic='pearson')  # default: no presence filter
    r, t, _ = stat.fit_transform(X=X, y=y, covariates=None, device=torch.device('cpu'))
    # Every column has variance > 0, so none is dropped by the variance gate.
    assert bool((t[:, 0].numpy() != 0.0).all()), "no edge should be filtered when off"


def test_presence_filter_adds_to_variance_gate():
    """An edge with variance but present in a minority is kept by the variance
    gate yet dropped by the presence filter (shows the filter is additive)."""
    rng = np.random.RandomState(1)
    n = 100
    y = rng.randn(n, 1).astype(np.float32)
    X = np.zeros((n, 1), dtype=np.float32)
    X[:20, 0] = rng.randn(20).astype(np.float32) + 5.0  # nonzero in 20% only

    no_filter = EdgeStatistic(selection_statistic='pearson')
    _, t_off, _ = no_filter.fit_transform(X=X, y=y, covariates=None, device=torch.device('cpu'))
    assert t_off[0, 0].item() != 0.0  # survives the variance gate

    with_filter = EdgeStatistic(selection_statistic='pearson', presence_filter=0.5)
    _, t_on, _ = with_filter.fit_transform(X=X, y=y, covariates=None, device=torch.device('cpu'))
    assert t_on[0, 0].item() == 0.0  # dropped by the presence filter


# --- Connected-component edge filtering ---------------------------------------

def test_resolve_min_component_size():
    assert resolve_min_component_size(False) is None
    assert resolve_min_component_size(None) is None
    assert resolve_min_component_size(True) == 2
    assert resolve_min_component_size(3) == 3
    with pytest.raises(ValueError):
        resolve_min_component_size(0)
    with pytest.raises(ValueError):
        resolve_min_component_size(-2)


def test_filter_connected_components_drops_lone_edges():
    """A lone edge is dropped; a connected chain of edges is kept.

    5 nodes -> 10 upper-triangular edges. Feature index 0 = (0,1), 4 = (1,2)
    (a 2-edge chain sharing node 1), and 9 = (3,4) (an isolated pair).
    """
    mask = torch.zeros(10, 2, 1, dtype=torch.bool)
    mask[[0, 4, 9], Networks.positive, 0] = True

    out = filter_connected_components(mask, min_edges=2)
    assert bool(out[0, Networks.positive, 0])       # chain edge kept
    assert bool(out[4, Networks.positive, 0])       # chain edge kept
    assert not bool(out[9, Networks.positive, 0])   # lone edge dropped


def _sel_with_edges(connected_components):
    sel = UnivariateEdgeSelection(
        selection_statistic='pearson',
        connected_components=connected_components,
        edge_selection=[PThreshold(threshold=[0.05], correction=[None])],
    )
    sel.set_params(**list(sel.param_grid)[0])
    # 5 nodes / 10 edges: positive edges 0=(0,1), 4=(1,2) form a chain; 9=(3,4)
    # is isolated. All are strongly, significantly positive.
    r = torch.zeros(10, 1)
    t = torch.zeros(10, 1)
    for e in (0, 4, 9):
        r[e, 0] = 0.5
        t[e, 0] = 6.0        # p ~ 1e-7 at df=100
    sel.r_edges, sel.t_edges, sel.df = r, t, 100
    return sel


def test_connected_components_filter_end_to_end():
    """With connected_components on, the isolated edge is not selected; off, it is."""
    edges_on = _sel_with_edges(True).return_selected_edges()
    assert bool(edges_on[0, Networks.positive, 0])
    assert bool(edges_on[4, Networks.positive, 0])
    assert not bool(edges_on[9, Networks.positive, 0])  # lone edge removed

    edges_off = _sel_with_edges(False).return_selected_edges()
    assert bool(edges_off[9, Networks.positive, 0])     # lone edge kept


def _t_and_exact_p(rng, n_features, n_runs, df):
    """Random t statistics spanning the selection boundary, with their exact p."""
    from scipy import stats
    t = rng.standard_t(df, size=(n_features, n_runs)) * 2.5
    return t, 2 * stats.t.sf(np.abs(t), df)


@pytest.mark.parametrize("correction", [None, "bonferroni", "sidak", "holm", "fdr_bh"])
def test_pthreshold_matches_statsmodels_per_run(correction):
    """Selection equals exact p-values, corrected by statsmodels one run at a time.

    The reference is independent of the code under test: scipy's exact t tail,
    then statsmodels' correction applied separately to each column.
    """
    from statsmodels.stats import multitest

    rng = np.random.RandomState(11)
    df, n_features, n_runs = 40, 200, 3
    t, p = _t_and_exact_p(rng, n_features, n_runs, df)
    r = np.sign(t) * 0.1
    threshold = 0.05

    p_corrected = p.copy()
    if correction is not None:
        for run in range(n_runs):
            _, p_corrected[:, run], _, _ = multitest.multipletests(
                p[:, run], alpha=0.05, method=correction)
    expected_pos = (p_corrected < threshold) & (r > 0)
    expected_neg = (p_corrected < threshold) & (r < 0)
    assert expected_pos.any() and not expected_pos.all()   # the case is not trivial

    selector = PThreshold(threshold=threshold, correction=correction)
    edges = selector.select(r=torch.as_tensor(r), t=torch.as_tensor(t), df=df)

    # select() returns [Features, 2, N_thresholds, Runs]; one threshold here.
    assert edges.shape == (n_features, 2, 1, n_runs)
    np.testing.assert_array_equal(edges[:, 0, 0].numpy(), expected_pos)
    np.testing.assert_array_equal(edges[:, 1, 0].numpy(), expected_neg)


@pytest.mark.parametrize("correction", ["bonferroni", "sidak", "holm", "fdr_bh"])
def test_a_runs_selection_does_not_depend_on_its_batch(correction):
    """A run is corrected over its own edges, not over every run in the batch.

    Regression test. The correction used to run over ``p.flatten()``, so a
    permutation chunk of shape [F, P] was corrected as F x P tests: each permuted
    run faced a far stricter threshold than the real run (and one that changed
    with the chunk size), making the permutation null weaker than the procedure
    it was meant to be a null of.
    """
    rng = np.random.RandomState(5)
    df, n_features, n_runs = 40, 150, 50
    t, _ = _t_and_exact_p(rng, n_features, n_runs, df)
    r = torch.as_tensor(np.sign(t) * 0.1)
    t = torch.as_tensor(t)

    selector = PThreshold(threshold=0.05, correction=correction)
    batched = selector.select(r=r, t=t, df=df)
    for run in (0, n_runs // 2, n_runs - 1):
        alone = selector.select(r=r[:, [run]], t=t[:, [run]], df=df)
        assert torch.equal(batched[..., [run]], alone), f"run {run} changed with its batch"


# --- Threshold-grid (params axis) tests --------------------------------


def test_select_params_axis_matches_per_threshold_calls():
    """PThreshold.select's params dim must reproduce independent
    select() calls, one per threshold, exactly."""
    rng = np.random.RandomState(21)
    r = torch.as_tensor(rng.randn(30, 4))
    t = torch.as_tensor(rng.randn(30, 4) * 4)
    thresholds = [0.01, 0.05, 0.1]

    selector = PThreshold(threshold=thresholds, correction='bonferroni')
    batched = selector.select(r=r, t=t, df=50, thresholds=thresholds)  # [F,2,3,4]
    assert batched.shape == (30, 2, 3, 4)

    for i, threshold in enumerate(thresholds):
        ref_selector = PThreshold(threshold=threshold, correction='bonferroni')
        ref = ref_selector.select(r=r, t=t, df=50)          # [F, 2, 1, R]
        assert torch.equal(batched[:, :, i, :], ref[:, :, 0, :])
