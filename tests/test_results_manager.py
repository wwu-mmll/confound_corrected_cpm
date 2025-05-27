import os
import unittest
import tempfile
import shutil
import numpy as np
import pandas as pd

from cpm.results_manager import ResultsManager, PermutationManager
from cpm.utils import vector_to_upper_triangular_matrix


class TestResultsManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_update_results_directory_permutation(self):
        mgr = ResultsManager(output_dir=self.temp_dir,
                             perm_run=1, n_folds=2, n_features=2)
        expected = os.path.join(self.temp_dir, 'permutation', '1')
        self.assertTrue(os.path.isdir(expected))
        self.assertEqual(mgr.results_directory, expected)

    def test_update_results_directory_base(self):
        mgr = ResultsManager(output_dir=self.temp_dir,
                             perm_run=0, n_folds=2, n_features=2)
        self.assertTrue(os.path.isdir(self.temp_dir))
        self.assertEqual(mgr.results_directory, self.temp_dir)

    def test_initialize_edges_without_params(self):
        e = ResultsManager.initialize_edges(4, 5)
        self.assertEqual(e['positive'].shape, (4, 5))
        self.assertEqual(e['negative'].shape, (4, 5))

    def test_initialize_edges_with_params(self):
        e = ResultsManager.initialize_edges(3, 6, n_params=2)
        self.assertEqual(e['positive'].shape, (3, 6, 2))
        self.assertEqual(e['negative'].shape, (3, 6, 2))

    def test_store_edges_and_calculate_stability(self):
        mgr = ResultsManager(output_dir=self.temp_dir,
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
            self.assertTrue(os.path.exists(edges_file))
            self.assertTrue(os.path.exists(stab_file))
            # loading should match triangular transform
            arr = np.load(edges_file)
            np.testing.assert_array_equal(arr, vector_to_upper_triangular_matrix(mgr.cv_edges[sign][0]))

    def test_load_cv_results_filters_mean(self):
        idx = pd.MultiIndex.from_product([[0, 1], ['pos', 'neg']], names=['fold', 'network'])
        cols = pd.MultiIndex.from_product([['full', 'covariates'], ['mean', 'std']])
        df = pd.DataFrame(np.random.rand(4, 4), index=idx, columns=cols)
        path = os.path.join(self.temp_dir, 'cv_results_mean_std.csv')
        df.to_csv(path)
        loaded = ResultsManager.load_cv_results(self.temp_dir)
        # Only mean columns remain
        self.assertEqual(set(loaded.columns), {'full', 'covariates'})
        self.assertEqual(loaded.shape, (4, 2))


class TestPermutationManager(unittest.TestCase):
    def test__calculate_group_p_value_score_and_error(self):
        true = pd.DataFrame({'spearman_score': [0.2], 'mse_error': [1.5]})
        perms = pd.DataFrame({'spearman_score': [0.1, 0.3, 0.25], 'mse_error': [1.4, 1.6, 1.5]})
        p = PermutationManager._calculate_group_p_value(true, perms)
        self.assertAlmostEqual(p['spearman_score'], 2 / 4)
        self.assertAlmostEqual(p['mse_error'], 1 / 4)

    def test_calculate_p_values_groups(self):
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
        self.assertEqual(set(pvals.index), {('pos', 'full'), ('neg', 'full')})
        self.assertIn('spearman_score', pvals.columns)
        self.assertIn('mse_error', pvals.columns)


if __name__ == '__main__':
    unittest.main()
