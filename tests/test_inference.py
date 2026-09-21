"""
Tests for the permutation inference in cccpm.inference.

These check the p-value definitions directly -- the +1 correction, the
direction flip for lower-is-better metrics, and the NBS/TFCE cluster statistics
against hand-constructed graphs whose component structure is known by
construction.
"""
import pytest
import numpy as np
import pandas as pd

from cccpm.inference import PermutationManager


class TestPermutationManager:
    def test_calculate_group_p_value_higher_is_better(self):
        """For metrics where higher is better, p = (count(true < perm) + 1) / (n_perms + 1)."""
        true = pd.DataFrame({'pearson_score': [0.5]})
        perms = pd.DataFrame({'pearson_score': [0.3, 0.6, 0.4, 0.7]})

        p = PermutationManager._calculate_group_p_value(true, perms)

        # true (0.5) < perm: 0.6, 0.7 → 2 out of 4. p = (2+1)/(4+1) = 0.6
        assert p['pearson_score'] == pytest.approx(3 / 5)

    def test_calculate_group_p_value_lower_is_better(self):
        """For error metrics (lower is better), p = (count(true > perm) + 1) / (n_perms + 1)."""
        true = pd.DataFrame({'mean_squared_error': [1.5]})
        perms = pd.DataFrame({'mean_squared_error': [1.4, 1.6, 1.5, 1.7]})

        p = PermutationManager._calculate_group_p_value(true, perms)

        # true (1.5) > perm: 1.4 → 1 out of 4. p = (1+1)/(4+1) = 0.4
        assert p['mean_squared_error'] == pytest.approx(2 / 5)

    def test_calculate_group_p_value_never_exceeds_one(self):
        """A valid p-value must be in (0, 1] even when every permutation beats the true value."""
        true = pd.DataFrame({'pearson_score': [0.0]})
        perms = pd.DataFrame({'pearson_score': [0.5, 0.6, 0.7]})  # all beat true

        p = PermutationManager._calculate_group_p_value(true, perms)

        # (3 + 1) / (3 + 1) = 1.0 — must not exceed 1
        assert p['pearson_score'] == pytest.approx(1.0)
        assert 0 < p['pearson_score'] <= 1

    def test_calculate_p_values_groups(self):
        """Test grouped p-value calculation across model/network combinations."""
        df_true = pd.DataFrame({
            'network': ['positive', 'negative'],
            'model': ['connectome', 'connectome'],
            'pearson_score': [0.5, 0.6],
            'mean_squared_error': [1.0, 0.9]
        }).set_index(['network', 'model'])

        perms_list = []
        for vals in ([0.4, 0.7], [0.6, 0.5]):
            df = pd.DataFrame({
                'network': ['positive', 'negative'],
                'model': ['connectome', 'connectome'],
                'pearson_score': vals,
                'mean_squared_error': [1.1, 0.8]
            }).set_index(['network', 'model'])
            perms_list.append(df)

        all_perms = pd.concat(perms_list)
        pvals = PermutationManager.calculate_p_values(df_true, all_perms)

        assert ('positive', 'connectome') in pvals.index
        assert ('negative', 'connectome') in pvals.index
        assert 'pearson_score' in pvals.columns
        assert 'mean_squared_error' in pvals.columns

    def test_is_lower_better(self):
        assert PermutationManager._is_lower_better('mean_squared_error') == True
        assert PermutationManager._is_lower_better('mean_absolute_error') == True
        assert PermutationManager._is_lower_better('pearson_score') == False
        assert PermutationManager._is_lower_better('accuracy') == False
        assert PermutationManager._is_lower_better('some_error') == True


def _make_stability_arrays(n_nodes, n_perms, clique, isolated, seed=0,
                           clique_val=0.9, isolated_val=0.8, null_density=0.05):
    """Build (true, permutation) stability arrays of the stored shape
    [n_nodes, n_nodes, 2, runs]. A dense positive-network clique and a few
    isolated positive-network edges are planted in the observed data; the null
    is sparse low-stability noise. The negative network is left empty."""
    rng = np.random.default_rng(seed)
    triu = np.triu_indices(n_nodes, k=1)

    true = np.zeros((n_nodes, n_nodes, 2, 1))
    for a in range(len(clique)):
        for b in range(a + 1, len(clique)):
            true[clique[a], clique[b], 0, 0] = clique_val
            true[clique[b], clique[a], 0, 0] = clique_val
    for (i, j) in isolated:
        true[i, j, 0, 0] = isolated_val
        true[j, i, 0, 0] = isolated_val

    perm = np.zeros((n_nodes, n_nodes, 2, n_perms))
    for p in range(n_perms):
        for layer in (0, 1):
            vals = (rng.random(len(triu[0])) < null_density) * rng.choice(
                [0.4, 0.6], len(triu[0]))
            perm[triu[0], triu[1], layer, p] = vals
            perm[triu[1], triu[0], layer, p] = vals
    return true, perm


class TestEdgeSignificance:
    def test_nbs_output_contract(self):
        """Shape, per-layer symmetry, and p-values in (0, 1] with the
        permutation floor at 1 / (n_perms + 1)."""
        n_perms = 100
        true, perm = _make_stability_arrays(20, n_perms, clique=range(6),
                                            isolated=[(10, 11)])
        sig = PermutationManager.calculate_p_values_edges_nbs(true, perm)
        assert sig.shape == (20, 20, 2)
        assert np.allclose(sig[:, :, 0], sig[:, :, 0].T)
        assert np.allclose(sig[:, :, 1], sig[:, :, 1].T)
        assert np.all(sig > 0) and np.all(sig <= 1)
        assert sig.min() >= 1.0 / (n_perms + 1) - 1e-12

    def test_nbs_detects_planted_subnetwork(self):
        true, perm = _make_stability_arrays(20, 200, clique=range(6),
                                            isolated=[(10, 11), (14, 17)])
        sig = PermutationManager.calculate_p_values_edges_nbs(
            true, perm, threshold=0.5, component_stat="extent")
        # every clique edge is significant; isolated weak edges are not
        assert sig[0, 1, 0] < 0.05
        assert sig[10, 11, 0] == 1.0
        # empty negative network -> all p == 1
        assert np.all(sig[:, :, 1] == 1.0)

    def test_nbs_extent_and_intensity_both_run(self):
        true, perm = _make_stability_arrays(20, 100, clique=range(6),
                                            isolated=[(10, 11)])
        ext = PermutationManager.calculate_p_values_edges_nbs(
            true, perm, component_stat="extent")
        inten = PermutationManager.calculate_p_values_edges_nbs(
            true, perm, component_stat="intensity")
        assert ext[0, 1, 0] < 0.05
        assert inten[0, 1, 0] < 0.05

    def test_nbs_rejects_unknown_component_stat(self):
        true, perm = _make_stability_arrays(10, 20, clique=range(4), isolated=[])
        with pytest.raises(ValueError):
            PermutationManager.calculate_p_values_edges_nbs(
                true, perm, component_stat="bogus")

    def test_nbs_deterministic(self):
        true, perm = _make_stability_arrays(20, 100, clique=range(6),
                                            isolated=[(10, 11)])
        a = PermutationManager.calculate_p_values_edges_nbs(true, perm)
        b = PermutationManager.calculate_p_values_edges_nbs(true, perm)
        assert np.array_equal(a, b)

    def test_tfce_shape_and_bounds(self):
        true, perm = _make_stability_arrays(20, 100, clique=range(6),
                                            isolated=[(10, 11)])
        sig = PermutationManager.calculate_p_values_edges_tfce(true, perm)
        assert sig.shape == (20, 20, 2)
        assert np.all(sig > 0) and np.all(sig <= 1)

    def test_tfce_strong_isolated_edge_can_be_significant(self):
        # A strongly-stable isolated edge (0.8) beats a sparse null capped at
        # 0.6, with no primary threshold needed. The null density is kept low so
        # it cannot form a large connected 0.6-stability cluster: extent-weighted
        # TFCE legitimately lets such a cluster outscore a single strong edge, so
        # a dense null would (correctly) mask the isolated edge.
        true, perm = _make_stability_arrays(20, 200, clique=range(6),
                                            isolated=[(10, 11)],
                                            isolated_val=0.8, null_density=0.02)
        sig = PermutationManager.calculate_p_values_edges_tfce(true, perm)
        assert sig[10, 11, 0] < 0.05

    def test_nbs_diagnostics(self):
        import json
        true, perm = _make_stability_arrays(20, 200, clique=range(6),
                                            isolated=[(10, 11)])
        sig, meta = PermutationManager.calculate_p_values_edges_nbs(
            true, perm, return_diagnostics=True)
        assert sig.shape == (20, 20, 2)
        assert meta["method"] == "nbs"
        assert meta["n_permutations"] == 200
        pos = meta["networks"]["positive"]
        assert len(pos["max_null"]) == 200
        assert pos["largest_component_edges"] == 15  # 6-node clique
        assert pos["n_significant_components"] >= 1
        assert pos["components"][0]["statistic"] >= pos["components"][-1]["statistic"]
        # JSON-serialisable (this is what gets written to disk)
        assert json.dumps(meta)

    def test_tfce_diagnostics(self):
        import json
        true, perm = _make_stability_arrays(20, 100, clique=range(6),
                                            isolated=[(10, 11)])
        sig, meta = PermutationManager.calculate_p_values_edges_tfce(
            true, perm, return_diagnostics=True)
        assert meta["method"] == "tfce"
        pos = meta["networks"]["positive"]
        assert len(pos["max_null"]) == 100
        assert pos["observed_max"] > 0
        assert json.dumps(meta)


class TestStableEdgesContext:
    def test_context_uncapped_with_csv_and_method_info(self, tmp_path):
        from cccpm.reporting.section_builders import build_stable_edges_context

        true, perm = _make_stability_arrays(20, 200, clique=range(8),
                                            isolated=[(10, 11)])
        sig, meta = PermutationManager.calculate_p_values_edges_nbs(
            true, perm, return_diagnostics=True)

        ctx = build_stable_edges_context(
            edge_stability=true,
            edge_stability_significance=sig,
            atlas_labels=None,
            significance_meta=meta,
            plots_dir=str(tmp_path),
        )
        assert ctx["has_edge_data"] is True
        assert ctx["edge_method"] == "nbs"
        assert "Network-Based Statistic" in ctx["edge_method_label"]
        # every significant clique edge shown (8-node clique = 28 edges), no cap
        assert ctx["edge_count_positive"] == 28
        assert ctx["edge_csv_data_uri"].startswith("data:text/csv;base64,")
        assert ctx["null_plot_positive"]  # figure embedded


# ============================================================
# The top-level entry point.
#
# calculate_permutation_results reads the true and permuted results back off
# disk and writes the p-value files the report reads. Until this test it was
# covered only indirectly, by the example scripts in test_integration.py --
# which is how a missing import survived a file split undetected for a full
# test run. This exercises it directly, in under a second.
# ============================================================

def _write_cv_summary(folder, n_runs, seed):
    """Write a cv_results_summary.csv in the two-level format the pipeline saves."""
    from cccpm.constants import Models, Networks

    rng = np.random.RandomState(seed)
    metrics = ['pearson_score', 'mean_squared_error']
    index = pd.MultiIndex.from_product(
        [[m.name for m in Models], [n.name for n in Networks], list(range(n_runs))],
        names=['model', 'network', 'run'],
    )
    df_mean = pd.DataFrame(rng.rand(len(index), len(metrics)), index=index, columns=metrics)
    df_std = pd.DataFrame(rng.rand(len(index), len(metrics)), index=index, columns=metrics)
    df = pd.concat([df_mean, df_std], axis=1, keys=['mean', 'std'])
    df = df.swaplevel(0, 1, axis=1).sort_index(axis=1)
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / 'cv_results_summary.csv')


def _permutation_results_dir(tmp_path, n_perms=8, n_nodes=10):
    """A minimal results directory: true run plus a permutation subdirectory."""
    rng = np.random.RandomState(0)
    perm_dir = tmp_path / 'permutation'
    _write_cv_summary(tmp_path, n_runs=1, seed=0)
    _write_cv_summary(perm_dir, n_runs=n_perms, seed=1)

    # Stability maps are stored as connectomes: [n_nodes, n_nodes, 2 (pos/neg), runs].
    def _symmetric(n_runs):
        a = rng.rand(n_nodes, n_nodes, 2, n_runs).astype(np.float32)
        a = (a + a.transpose(1, 0, 2, 3)) / 2
        for r in range(n_runs):
            for layer in range(2):
                np.fill_diagonal(a[:, :, layer, r], 0.0)
        return a

    np.save(tmp_path / 'stability_edges.npy', _symmetric(1))
    np.save(perm_dir / 'stability_edges.npy', _symmetric(n_perms))
    return tmp_path


@pytest.mark.parametrize("method", ["nbs", "tfce"])
def test_calculate_permutation_results_writes_all_outputs(tmp_path, method):
    import logging

    results_dir = _permutation_results_dir(tmp_path)

    PermutationManager.calculate_permutation_results(
        str(results_dir), logging.getLogger(__name__), method=method)

    p_values = pd.read_csv(results_dir / 'p_values.csv', index_col=[0, 1])
    assert ((p_values.to_numpy() >= 0) & (p_values.to_numpy() <= 1)).all()

    sig = np.load(results_dir / 'stability_edges_significance.npy')
    assert sig.shape[:2] == (10, 10)
    assert ((sig >= 0) & (sig <= 1)).all()

    assert (results_dir / 'stability_edges_significance_meta.json').exists()


def test_calculate_permutation_results_rejects_unknown_method(tmp_path):
    import logging

    results_dir = _permutation_results_dir(tmp_path)
    with pytest.raises(ValueError, match="Unknown edge-significance method"):
        PermutationManager.calculate_permutation_results(
            str(results_dir), logging.getLogger(__name__), method='bogus')


# ============================================================
# Undefined statistics have no null distribution.
# ============================================================

def test_p_value_is_nan_when_the_observed_statistic_is_undefined():
    """A NaN observed value must give a NaN p-value, not the permutation floor.

    Every comparison with NaN is False, so the naive count comes out 0 and the
    +1 correction reports 1/(n_perms+1) -- the *most* significant p-value the
    test can produce -- for a statistic that does not exist. A run without
    covariates hits this directly: its covariates/full/residuals/increment rows
    are NaN placeholders, and they were being reported as p = 0.02.
    """
    true = pd.DataFrame({'pearson_score': [float('nan')]})
    perms = pd.DataFrame({'pearson_score': [float('nan')] * 50})

    p = PermutationManager._calculate_group_p_value(true, perms)

    assert np.isnan(p['pearson_score'])


def test_p_value_is_nan_when_only_the_null_is_undefined():
    """Defensive: a real observation against an all-NaN null is still undefined."""
    true = pd.DataFrame({'pearson_score': [0.4]})
    perms = pd.DataFrame({'pearson_score': [float('nan')] * 50})

    p = PermutationManager._calculate_group_p_value(true, perms)

    assert np.isnan(p['pearson_score'])
