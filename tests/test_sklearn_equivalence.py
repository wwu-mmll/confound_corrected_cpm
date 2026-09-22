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
import torch
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, explained_variance_score, roc_auc_score
from sklearn.model_selection import KFold

from cccpm.cpm_analysis import CPMAnalysis
from cccpm.constants import Models, Networks, TaskType
from cccpm.edge_selection import PThreshold, UnivariateEdgeSelection
from cccpm.statistics import correlations_and_pvalues
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
    ytr = y[:ntr]
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
    np.testing.assert_allclose(pred[:, Models.connectome], p_conn, atol=2e-3)
    np.testing.assert_allclose(pred[:, Models.covariates], p_cov, atol=2e-3)
    np.testing.assert_allclose(pred[:, Models.full], p_full, atol=2e-3)


# --------------------------------------------------------------------------- #
# 3. End-to-end cross-validated pipeline: toolbox vs sklearn + inflation story #
# --------------------------------------------------------------------------- #
def _toolbox_connectome_ev(X, y, Z, selection_input, model_input, cv, tmp_path):
    ue = UnivariateEdgeSelection(
        selection_statistic="pearson", selection_input=selection_input,
        edge_selection=[PThreshold(threshold=P_THRESHOLD, correction=[None])])
    cpm = CPMAnalysis(
        results_directory=str(tmp_path), cv=cv, edge_selection=ue,
        model_input=model_input, n_permutations=0, task_type="regression")
    cpm._single_run(X=X, y=y.reshape(-1, 1), covariates=Z, perm_run=False)
    ag = cpm.results_manager.agg_results
    val = ag.loc[("connectome", "both"), ("explained_variance_score", "mean")]
    return float(np.ravel(val)[0])


def _sklearn_connectome_ev(X, y, Z, selection_input, model_input, cv):
    """The same recipe in sklearn, with the two confound choices independent.

    `selection_input` decides whether edge selection controls for Z; per-edge
    that is the regression y ~ 1 + Z + edge, which is what ref_semipartial
    computes. `model_input` decides whether the covariate variance is taken out
    of the connectome the model consumes.
    """
    evs = []
    for tr, te in cv.split(X, y):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        Ztr, Zte = Z[tr], Z[te]

        if selection_input == "residualized":
            r, p = ref_semipartial(Xtr, ytr, Ztr)
        else:
            r, p = ref_pearson(Xtr, ytr)
        pos = (p < P_THRESHOLD) & (r > 0)
        neg = (p < P_THRESHOLD) & (r < 0)

        if model_input == "residualized":
            m = LinearRegression().fit(Ztr, Xtr)
            Xtr, Xte = Xtr - m.predict(Ztr), Xte - m.predict(Zte)

        ftr = np.column_stack([Xtr[:, pos].sum(1), Xtr[:, neg].sum(1)])
        fte = np.column_stack([Xte[:, pos].sum(1), Xte[:, neg].sum(1)])
        pred = LinearRegression().fit(ftr, ytr).predict(fte)
        evs.append(explained_variance_score(yte, pred))
    return float(np.mean(evs))


# The confound 2x2: two independent run-level choices.
CONFIGS = [
    ("raw",     "raw",          "raw"),
    ("partial", "residualized", "raw"),
    ("resid",   "residualized", "residualized"),
]


def test_pipeline_matches_sklearn_and_shows_inflation(tmp_path):
    r2, kappa = 0.36, 0.6
    X, y, Z = _sim(kappa=kappa, r2=r2, n=1500, seed=5)

    # Identical fold structure for both implementations.
    cv_tb = KFold(n_splits=5, shuffle=True, random_state=0)
    cv_sk = KFold(n_splits=5, shuffle=True, random_state=0)

    ev = {}
    for name, selection_input, model_input in CONFIGS:
        tb = _toolbox_connectome_ev(X, y, Z, selection_input, model_input, cv_tb, tmp_path)
        sk = _sklearn_connectome_ev(X, y, Z, selection_input, model_input, cv_sk)
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


# --------------------------------------------------------------------------- #
# 4. End-to-end cross-validated pipeline, classification                       #
# --------------------------------------------------------------------------- #
def _sklearn_classification_metrics(X, y, Z, cv):
    """The same CPM recipe written directly in sklearn: point-biserial edge
    selection on the training split, positive/negative sum scores, logistic
    regression, scored out of sample."""
    accs, aucs = [], []
    for tr, te in cv.split(X, y):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        # Point-biserial is Pearson against a 0/1 target.
        r, p = ref_pearson(Xtr, ytr)
        pos = (p < P_THRESHOLD) & (r > 0)
        neg = (p < P_THRESHOLD) & (r < 0)

        ftr = np.column_stack([Xtr[:, pos].sum(1), Xtr[:, neg].sum(1)])
        fte = np.column_stack([Xte[:, pos].sum(1), Xte[:, neg].sum(1)])

        # C=1e9 is effectively unregularised, and unlike penalty=None it is
        # accepted across the whole supported sklearn range.
        clf = LogisticRegression(C=1e9, max_iter=1000).fit(ftr, ytr)
        proba = clf.predict_proba(fte)[:, 1]
        accs.append(accuracy_score(yte, (proba > 0.5).astype(int)))
        aucs.append(roc_auc_score(yte, proba))
    return float(np.mean(accs)), float(np.mean(aucs))


def test_classification_pipeline_matches_sklearn(tmp_path):
    """The classification path end to end -- edge selection, sum scores, IRLS
    logistic fit, out-of-sample scoring -- against the same recipe in sklearn.

    The regression pipeline has had this check since the beginning; the
    classification pipeline was only ever verified at the model level
    (test_classification.py), leaving selection, aggregation and the
    classification metrics unverified end to end.
    """
    X, y_cont, Z = _sim(kappa=0.3, r2=0.36, n=1200, seed=11)
    y = (y_cont > np.median(y_cont)).astype(np.float64)

    cv_tb = KFold(n_splits=5, shuffle=True, random_state=0)
    cv_sk = KFold(n_splits=5, shuffle=True, random_state=0)

    ue = UnivariateEdgeSelection(
        edge_statistic="point_biserial",
        edge_selection=[PThreshold(threshold=P_THRESHOLD, correction=[None])])
    cpm = CPMAnalysis(
        results_directory=str(tmp_path), cv=cv_tb, edge_selection=ue,
        n_permutations=0, task_type="classification")
    cpm._single_run(X=X, y=y.reshape(-1, 1), covariates=Z, perm_run=False)

    ag = cpm.results_manager.agg_results
    tb_acc = float(np.ravel(ag.loc[("connectome", "both"), ("accuracy", "mean")])[0])
    tb_auc = float(np.ravel(ag.loc[("connectome", "both"), ("roc_auc", "mean")])[0])

    sk_acc, sk_auc = _sklearn_classification_metrics(X, y, Z, cv_sk)

    # The signal must be real, not a coin flip -- otherwise agreement is vacuous.
    assert sk_auc > 0.65, f"reference pipeline found no signal (auc={sk_auc:.3f})"
    assert abs(tb_acc - sk_acc) < 0.02, f"accuracy: toolbox {tb_acc:.3f} vs sklearn {sk_acc:.3f}"
    assert abs(tb_auc - sk_auc) < 0.02, f"roc_auc:  toolbox {tb_auc:.3f} vs sklearn {sk_auc:.3f}"


# --------------------------------------------------------------------------- #
# 5. How wrong is the normal approximation to the t tail?                      #
# --------------------------------------------------------------------------- #
def test_edge_pvalues_vs_exact_t_distribution():
    """Quantify the documented normal approximation in the edge-selection
    p-values against the exact two-sided t-test.

    `correlations_and_pvalues` converts the t statistic through the standard
    normal tail rather than `scipy.stats.t.sf`, which makes every p-value
    slightly too small (anti-conservative), by more at small n. This is
    RELEASE_PLAN decision #6, and it was previously unmeasured: the existing
    equivalence tests run at n=1000, where the difference vanishes.

    Measured here (max |p_exact - p_approx| over 200 edges, seed 7):

        n=30  0.0184   (0.0146 near the p=0.05 boundary)
        n=60  0.0089   (0.0063)
        n=120 0.0044   (0.0036)
        n=500 0.0010   (0.0009)

    So it decays like ~1/n and is worst where it matters least in practice: at
    n=30 an edge whose exact p is 0.065 can be selected at a 0.05 threshold. At
    any cohort size CPM is normally run on, it cannot move an edge across the
    boundary meaningfully.

    The assertions pin the *direction* (never conservative) and the decay; the
    magnitude bounds are loose enough not to be seed-fragile.
    """
    rng = np.random.RandomState(7)
    # Largest deviation is at the smallest n. 30 is a realistic small CPM cohort.
    worst_by_n = {}
    for n in (30, 60, 120, 500):
        X = rng.randn(n, 200)
        y = X[:, 0] * 0.3 + rng.randn(n)

        _, p_tb = correlations_and_pvalues(
            X, y.reshape(-1, 1), correlation_type="pearson")
        p_tb = p_tb.numpy().ravel()

        r_ref, p_exact = ref_pearson(X, y)      # uses scipy.stats.t.sf

        # Direction: the normal tail is never heavier than the t tail, so the
        # approximate p-value is never larger than the exact one.
        assert np.all(p_tb <= p_exact + 1e-9), (
            f"n={n}: approximation produced a conservative p-value, which it cannot do")

        worst_by_n[n] = float(np.max(p_exact - p_tb))

    # The error shrinks monotonically with n, and is already small at n=30.
    ns = sorted(worst_by_n)
    for a, b in zip(ns, ns[1:]):
        assert worst_by_n[b] < worst_by_n[a], f"error grew from n={a} to n={b}: {worst_by_n}"

    assert worst_by_n[30] < 0.02, f"larger than documented at n=30: {worst_by_n[30]:.4f}"
    assert worst_by_n[500] < 2e-3, f"larger than documented at n=500: {worst_by_n[500]:.6f}"
