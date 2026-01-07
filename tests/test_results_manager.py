import os
import pytest
import numpy as np
import pandas as pd

from cccpm.results_manager import ResultsManager, PermutationManager
from cccpm.utils import vector_to_upper_triangular_matrix


def test_update_results_directory_permutation(tmp_path):
    # tmp_path is a Path object, convert to str for compatibility if needed
    tmp_str = str(tmp_path)

    mgr = ResultsManager(output_dir=tmp_str,
                         perm_run=1, n_folds=2, n_features=2)

    expected = os.path.join(tmp_str, 'permutation', '1')

    assert os.path.isdir(expected)
    assert mgr.results_directory == expected


def test_update_results_directory_base(tmp_path):
    tmp_str = str(tmp_path)

    mgr = ResultsManager(output_dir=tmp_str,
                         perm_run=0, n_folds=2, n_features=2)

    assert os.path.isdir(tmp_str)
    assert mgr.results_directory == tmp_str


def test_initialize_edges_without_params():
    e = ResultsManager.initialize_edges(4, 5)
    assert e['positive'].shape == (4, 5)
    assert e['negative'].shape == (4, 5)


def test_initialize_edges_with_params():
    e = ResultsManager.initialize_edges(3, 6, n_params=2)
    assert e['positive'].shape == (3, 6, 2)
    assert e['negative'].shape == (3, 6, 2)


def test_store_edges_and_calculate_stability(tmp_path):
    tmp_str = str(tmp_path)
    mgr = ResultsManager(output_dir=tmp_str,
                         perm_run=0, n_folds=2, n_features=3)

    mgr.store_edges({'positive': [0, 2], 'negative': [1]}, fold=0)
    mgr.store_edges({'positive': [1], 'negative': [0]}, fold=1)

    stability = mgr.calculate_edge_stability()

    # expected stability positive
    expected = np.sum(mgr.cv_edges['positive'], axis=0) / 2
    np.testing.assert_allclose(stability['positive'], expected)

    for sign in ['positive', 'negative']:
        edges_file = os.path.join(mgr.results_directory, f"{sign}_edges.npy")
        stab_file = os.path.join(mgr.results_directory, f"stability_{sign}_edges.npy")

        assert os.path.exists(edges_file)
        assert os.path.exists(stab_file)

        # loading should match triangular transform
        arr = np.load(edges_file)
        np.testing.assert_array_equal(
            arr,
            vector_to_upper_triangular_matrix(mgr.cv_edges[sign][0])
        )


def test_load_cv_results_filters_mean(tmp_path):
    idx = pd.MultiIndex.from_product([[0, 1], ['pos', 'neg']], names=['fold', 'network'])
    cols = pd.MultiIndex.from_product([['full', 'covariates'], ['mean', 'std']])
    df = pd.DataFrame(np.random.rand(4, 4), index=idx, columns=cols)

    path = tmp_path / 'cv_results_mean_std.csv'
    df.to_csv(path)

    loaded = ResultsManager.load_cv_results(str(tmp_path))

    # Only mean columns remain
    assert set(loaded.columns) == {'full', 'covariates'}
    assert loaded.shape == (4, 2)


# PermutationManager Tests
# (These were static methods, so they didn't need much setup change)

def test__calculate_group_p_value_score_and_error():
    true = pd.DataFrame({'spearman_score': [0.2], 'mse_error': [1.5]})
    perms = pd.DataFrame({'spearman_score': [0.1, 0.3, 0.25], 'mse_error': [1.4, 1.6, 1.5]})

    p = PermutationManager._calculate_group_p_value(true, perms)

    # assertAlmostEqual (unittest) -> pytest approx
    assert p['spearman_score'] == pytest.approx(2 / 4)
    assert p['mse_error'] == pytest.approx(1 / 4)


def test_calculate_p_values_groups():
    df_true = pd.DataFrame({
        'network': ['pos', 'neg'],
        'model': ['full', 'full'],
        'spearman_score': [0.5, 0.6],
        'mse_error': [1.0, 0.9]
    }).set_index(['network', 'model'])

    perms_list = []
    for vals in ([0.4, 0.7], [0.6, 0.5]):
        df = pd.DataFrame({
            'network': ['pos', 'neg'],
            'model': ['full', 'full'],
            'spearman_score': vals,
            'mse_error': [1.1, 0.8]
        }).set_index(['network', 'model'])
        perms_list.append(df)

    all_perms = pd.concat(perms_list)
    pvals = PermutationManager.calculate_p_values(df_true, all_perms)

    assert set(pvals.index) == {('pos', 'full'), ('neg', 'full')}
    assert 'spearman_score' in pvals.columns
    assert 'mse_error' in pvals.columns