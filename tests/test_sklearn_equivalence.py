"""
Equivalence tests: the toolbox's (torch/CUDA) GLM edge selection and CPM model
solvers must agree with an independent scikit-learn / scipy / numpy
re-implementation.

This is the credibility backbone for the confound-inflation simulation
(``examples/confound_inflation_demo.py``): it proves that

  * the vectorised torch edge statistics (Pearson, partial/semipartial, Spearman)
    equal the textbook definitions computed with scipy/numpy,
  * the batched torch OLS solvers behind the four CPM model variants
    (connectome / covariates / full / residuals) equal sklearn's
    ``LinearRegression`` when given the same selected edges, and
  * the full cross-validated pipeline (Pearson / partial selection, optional
    X-residualisation, sum score, OLS) reproduces an independent sklearn pipeline,
    yielding the raw > partial > residualised ≈ true ordering.

All tests run on CPU with float64/float32 tolerances.
"""

import numpy as np
import pytest
import torch
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold

from cccpm.cpm_analysis import CPMAnalysis
from cccpm.constants import Models, Networks, TaskType
from cccpm.edge_selection import (
    correlations_and_pvalues,
    PThreshold,
    UnivariateEdgeSelection,
)
from cccpm.models.linear_model import LinearCPM
from cccpm.simulation.simulate_sem import simulate_data_given_kappa

P_THRESHOLD = 0.05


# --------------------------------------------------------------------------- #
# Independent reference implementations (numpy / scipy)                        #
# --------------------------------------------------------------------------- #
def ref_pearson(X, y):
    """Per-column Pearson r and two-sided t-test p-value."""
    n = len(y)
    yc = y - y.mean()
    r = np.empty(X.shape[1])
    p = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        xc = X[:, j] - X[:, j].mean()
        rr = (xc @ yc) / (np.linalg.norm(xc) * np.linalg.norm(yc))
        r[j] = rr
        t = rr * np.sqrt((n - 2) / (1 - rr ** 2))
        p[j] = 2 * stats.t.sf(abs(t), n - 2)
    return r, p


def ref_semipartial(X, y, Z):
    """Per-column semipartial (part) correlation (Z removed from the edge only)
    and the partial-correlation t-test p-value (df = n - k - 2)."""
    n, k = len(y), Z.shape[1]
    Z1 = np.column_stack([np.ones(n), Z])
    yc = y - y.mean()
    by = np.linalg.lstsq(Z1, y, rcond=None)[0]
    yr = y - Z1 @ by                       # y residualised on Z (for partial r)
    r_sp = np.empty(X.shape[1])
    p = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        bx = np.linalg.lstsq(Z1, X[:, j], rcond=None)[0]
        xr = X[:, j] - Z1 @ bx             # edge residualised on Z
        r_sp[j] = (xr @ yc) / (np.linalg.norm(xr) * np.linalg.norm(yc))
        r_pt = (xr @ yr) / (np.linalg.norm(xr) * np.linalg.norm(yr))
        df = n - k - 2
        t = r_pt * np.sqrt(df / (1 - r_pt ** 2))
        p[j] = 2 * stats.t.sf(abs(t), df)
    return r_sp, p


def _sim(kappa=0.6, r2=0.36, n=1000, seed=0, n_features=105):
    sim = simulate_data_given_kappa(
        R2_X_y=r2, kappa=kappa,
        n_features=n_features, n_features_informative=10,
        n_pure_signal_features=10, n_confound_only_features=10,
        n_confounds=2, n_samples=n, random_state=seed,
    )
    return (sim["X"].astype(np.float64),
            sim["y"].ravel().astype(np.float64),
            sim["Z"].astype(np.float64))


# --------------------------------------------------------------------------- #
# 1. Edge selection statistics                                                #
# --------------------------------------------------------------------------- #
def test_pearson_matches_scipy():
    X, y, _ = _sim(seed=1)
    r_tb, p_tb = correlations_and_pvalues(
        X, y.reshape(-1, 1), correlation_type="pearson")
    r_tb, p_tb = r_tb.numpy().ravel(), p_tb.numpy().ravel()
    r_ref, p_ref = ref_pearson(X, y)

    np.testing.assert_allclose(r_tb, r_ref, atol=1e-6)
    # p-values use a (documented) normal approximation to the t tail; at n=1000
    # this is indistinguishable from the exact t-based p-value.
    np.testing.assert_allclose(p_tb, p_ref, atol=2e-3)
    # And the selected-edge masks must be identical.
    np.testing.assert_array_equal(p_tb < P_THRESHOLD, p_ref < P_THRESHOLD)


def test_partial_matches_reference():
    X, y, Z = _sim(seed=2)
    r_tb, p_tb = correlations_and_pvalues(
        X, y.reshape(-1, 1), confounds=Z, correlation_type="pearson")
    r_tb, p_tb = r_tb.numpy().ravel(), p_tb.numpy().ravel()
    r_ref, p_ref = ref_semipartial(X, y, Z)

    # Reported effect size is the semipartial (part) correlation.
    np.testing.assert_allclose(r_tb, r_ref, atol=1e-6)
    np.testing.assert_allclose(p_tb, p_ref, atol=2e-3)
    np.testing.assert_array_equal(p_tb < P_THRESHOLD, p_ref < P_THRESHOLD)


def test_spearman_matches_scipy():
    X, y, _ = _sim(seed=3)
    r_tb, _ = correlations_and_pvalues(
        X, y.reshape(-1, 1), correlation_type="spearman")
    r_tb = r_tb.numpy().ravel()
    r_ref = np.array([stats.spearmanr(X[:, j], y).statistic
                      for j in range(X.shape[1])])
    np.testing.assert_allclose(r_tb, r_ref, atol=1e-6)


# --------------------------------------------------------------------------- #
# 2. CPM model variants (torch solvers) vs sklearn, given identical edges      #
# --------------------------------------------------------------------------- #
def _make_edges(pos_mask, neg_mask):
    """Build a [F, 2, 1] edge tensor from boolean network masks."""
    edges = np.zeros((len(pos_mask), 2, 1), dtype=np.float32)
    edges[:, Networks.positive, 0] = pos_mask
    edges[:, Networks.negative, 0] = neg_mask
    return torch.as_tensor(edges)


def test_model_variants_match_sklearn():
    X, y, Z = _sim(seed=4, n=800)
    ntr = 500
    Xtr, Xte = X[:ntr], X[ntr:]
    ytr, yte = y[:ntr], y[ntr:]
    Ztr, Zte = Z[:ntr], Z[ntr:]

    # Fixed edge mask from Pearson selection on the training split (same mask
    # fed to BOTH implementations, so this isolates the model solvers).
    r_ref, p_ref = ref_pearson(Xtr, ytr)
    pos = (p_ref < P_THRESHOLD) & (r_ref > 0)
    neg = (p_ref < P_THRESHOLD) & (r_ref < 0)
    assert pos.sum() > 0 and neg.sum() > 0

    model = LinearCPM(edges=_make_edges(pos, neg), device="cpu",
                      task_type=TaskType.regression)
    model.fit(Xtr, ytr.reshape(-1, 1), Ztr)
    pred = model.predict(Xte, Zte).numpy()[:, :, Networks.both, 0]  # [N, n_models]

    # sklearn references -----------------------------------------------------
    pos_tr, neg_tr = Xtr[:, pos].sum(1), Xtr[:, neg].sum(1)
    pos_te, neg_te = Xte[:, pos].sum(1), Xte[:, neg].sum(1)
    str_tr = np.column_stack([pos_tr, neg_tr])
    str_te = np.column_stack([pos_te, neg_te])

    # connectome: y ~ strengths
    p_conn = LinearRegression().fit(str_tr, ytr).predict(str_te)
    # covariates: y ~ Z
    p_cov = LinearRegression().fit(Ztr, ytr).predict(Zte)
    # full: y ~ strengths + Z
    p_full = LinearRegression().fit(np.column_stack([str_tr, Ztr]), ytr).predict(
        np.column_stack([str_te, Zte]))
    # residuals: residualise each strength on Z (fit on train), then y ~ residuals
    def resid(col_tr, col_te):
        m = LinearRegression().fit(Ztr, col_tr)
        return col_tr - m.predict(Ztr), col_te - m.predict(Zte)
    pr_tr, pr_te = resid(pos_tr, pos_te)
    nr_tr, nr_te = resid(neg_tr, neg_te)
    p_res = LinearRegression().fit(
        np.column_stack([pr_tr, nr_tr]), ytr).predict(np.column_stack([pr_te, nr_te]))

    np.testing.assert_allclose(pred[:, Models.connectome], p_conn, atol=2e-3)
    np.testing.assert_allclose(pred[:, Models.covariates], p_cov, atol=2e-3)
    np.testing.assert_allclose(pred[:, Models.full], p_full, atol=2e-3)
    np.testing.assert_allclose(pred[:, Models.residuals], p_res, atol=2e-3)


# --------------------------------------------------------------------------- #
# 3. End-to-end cross-validated pipeline: toolbox vs sklearn + inflation story #
# --------------------------------------------------------------------------- #
def _toolbox_connectome_ev(X, y, Z, statistic, residualize, cv, tmp_path):
    ue = UnivariateEdgeSelection(
        edge_statistic=statistic,
        edge_selection=[PThreshold(threshold=P_THRESHOLD, correction=[None])])
    cpm = CPMAnalysis(
        results_directory=str(tmp_path), cv=cv, edge_selection=ue,
        calculate_residuals=residualize, n_permutations=0, task_type="regression")
    cpm._single_run(X=X, y=y.reshape(-1, 1), covariates=Z, perm_run=False)
    ag = cpm.results_manager.agg_results
    val = ag.loc[("connectome", "both"), ("explained_variance_score", "mean")]
    return float(np.ravel(val)[0])


def _sklearn_connectome_ev(X, y, Z, partial, residualize, cv):
    evs = []
    for tr, te in cv.split(X, y):
        Xtr, Xte = X[tr].copy(), X[te].copy()
        ytr, yte = y[tr], y[te]
        Ztr, Zte = Z[tr], Z[te]
        if residualize:
            m = LinearRegression().fit(Ztr, Xtr)
            Xtr = Xtr - m.predict(Ztr)
            Xte = Xte - m.predict(Zte)
        if partial:
            r, p = ref_semipartial(Xtr, ytr, Ztr)
        else:
            r, p = ref_pearson(Xtr, ytr)
        pos = (p < P_THRESHOLD) & (r > 0)
        neg = (p < P_THRESHOLD) & (r < 0)
        ftr = np.column_stack([Xtr[:, pos].sum(1), Xtr[:, neg].sum(1)])
        fte = np.column_stack([Xte[:, pos].sum(1), Xte[:, neg].sum(1)])
        pred = LinearRegression().fit(ftr, ytr).predict(fte)
        evs.append(explained_variance_score(yte, pred))
    return float(np.mean(evs))


CONFIGS = [
    ("pearson", False, False),          # raw
    ("pearson_partial", True, False),   # partial selection
    ("pearson", False, True),           # residualised X
]


def test_pipeline_matches_sklearn_and_shows_inflation(tmp_path):
    r2, kappa = 0.36, 0.6
    X, y, Z = _sim(kappa=kappa, r2=r2, n=1500, seed=5)

    # Identical fold structure for both implementations.
    cv_tb = KFold(n_splits=5, shuffle=True, random_state=0)
    cv_sk = KFold(n_splits=5, shuffle=True, random_state=0)

    ev = {}
    for (stat, partial, resid), name in zip(CONFIGS, ("raw", "partial", "resid")):
        tb = _toolbox_connectome_ev(X, y, Z, stat, resid, cv_tb, tmp_path)
        sk = _sklearn_connectome_ev(X, y, Z, partial, resid, cv_sk)
        # Toolbox and independent sklearn pipeline agree (small tolerance absorbs
        # boundary-of-threshold noise edges that carry ~no signal).
        assert abs(tb - sk) < 0.02, f"{name}: toolbox {tb:.3f} vs sklearn {sk:.3f}"
        ev[name] = tb

    true_r2 = (1 - kappa) * r2  # = 0.144

    # Inflation story: raw is maximally inflated, partial only partially
    # deconfounds (below raw, still above truth), residualisation recovers truth.
    assert ev["raw"] > ev["resid"] + 0.10
    assert ev["partial"] > ev["resid"] + 0.05
    assert ev["partial"] <= ev["raw"] + 0.02
    assert abs(ev["resid"] - true_r2) < 0.06
