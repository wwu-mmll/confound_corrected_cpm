"""Turns a null distribution into a p-value: model-level CV-metric permutation test, plus NBS/TFCE edge-stability correction.

The permutation *results* are produced upstream by `CPMAnalysis`; this module
only reads the true-vs-null arrays back off disk and does the statistics.
"""
import os
import json

import numpy as np
import pandas as pd

import networkx as nx

from cccpm.constants import Networks
from cccpm.results_manager import ResultsManager


class PermutationManager:
    """Permutation-based significance testing for CV metrics and edge stability."""

    @staticmethod
    def calculate_p_values(true_results, perms):
        """Returns a (network, model)-indexed DataFrame of p-values from true vs. permutation CV results."""
        grouped_true = true_results.groupby(['network', 'model'])
        grouped_perms = perms.groupby(['network', 'model'])

        p_values = []
        for (name, true_group), (_, perms_group) in zip(grouped_true, grouped_perms):
            p_value_series = PermutationManager._calculate_group_p_value(true_group, perms_group)
            p_values.append(pd.DataFrame(p_value_series).T.assign(network=name[0], model=name[1]))

        p_values_df = pd.concat(p_values).reset_index(drop=True)
        p_values_df = p_values_df.set_index(['network', 'model'])
        return p_values_df

    LOWER_IS_BETTER = {'mean_squared_error', 'mean_absolute_error'}

    @staticmethod
    def _is_lower_better(column_name):
        if column_name in PermutationManager.LOWER_IS_BETTER:
            return True
        if column_name.endswith('error'):
            return True
        return False

    @staticmethod
    def _calculate_group_p_value(true_group, perms_group):
        """Returns a Series of p-values for one (network, model) group of metrics."""
        result_dict = {}
        for column in true_group.columns:
            observed = true_group[column].values[0]
            null = perms_group[column].astype(float)

            # NaN compares False everywhere, so skipping this would floor the
            # p-value at 1/(n_perms+1) for a model that doesn't exist (e.g.
            # covariates/full/increment rows in a run without covariates).
            if pd.isna(observed) or null.isna().all():
                result_dict[column] = float('nan')
                continue

            if PermutationManager._is_lower_better(column):
                condition_count = (observed > null).sum()
            else:
                condition_count = (observed < null).sum()

            # Phipson & Smyth (2010): +1 in both terms counts the observed
            # statistic itself and guarantees a valid p-value in (0, 1].
            result_dict[column] = (condition_count + 1) / (len(null) + 1)

        return pd.Series(result_dict)

    @staticmethod
    def calculate_permutation_results(results_directory, logger, method="nbs",
                                      nbs_stability_threshold=0.5,
                                      nbs_component_stat="extent"):
        """Computes and saves model-level metric p-values and NBS/TFCE edge-stability significance to disk.

        Both edge methods control the family-wise error rate via a
        permutation max-statistic and write a per-edge p-value matrix to
        ``stability_edges_significance.npy`` (NBS: subnetwork-level, edges
        inherit their component's p-value; TFCE: per-edge).
        """
        true_results = ResultsManager.load_cv_results(results_directory)

        perm_dir = os.path.join(results_directory, 'permutation')
        perm_results = ResultsManager.load_cv_results(perm_dir)

        true_edge_stability = np.load(os.path.join(results_directory, 'stability_edges.npy'))
        perm_edge_stability = np.load(os.path.join(perm_dir, 'stability_edges.npy'))

        p_values = PermutationManager.calculate_p_values(true_results, perm_results)
        p_values.to_csv(os.path.join(results_directory, 'p_values.csv'))

        if method == "nbs":
            stability_significance, sig_meta = PermutationManager.calculate_p_values_edges_nbs(
                true_edge_stability, perm_edge_stability,
                threshold=nbs_stability_threshold,
                component_stat=nbs_component_stat,
                return_diagnostics=True)
        elif method == "tfce":
            stability_significance, sig_meta = PermutationManager.calculate_p_values_edges_tfce(
                true_edge_stability, perm_edge_stability, return_diagnostics=True)
        else:
            raise ValueError(
                f"Unknown edge-significance method '{method}'. Use 'nbs' or 'tfce'.")

        np.save(os.path.join(results_directory, 'stability_edges_significance.npy'), stability_significance)
        with open(os.path.join(results_directory, 'stability_edges_significance_meta.json'), 'w', encoding='utf-8') as f:
            json.dump(sig_meta, f)

        logger.debug("Saving significance of edge stability.")
        logger.info("Permutation test results")
        logger.info(p_values.round(4).to_string())
        return

    @staticmethod
    def _layer_arrays(true_stability, permutation_stability, layer):
        """Returns (observed [n_nodes, n_nodes], null [n_nodes, n_nodes, n_perms]) for one network layer (0=pos, 1=neg)."""
        true_layer = true_stability[:, :, layer, 0]
        perm_layer = permutation_stability[:, :, layer, :]
        return true_layer, perm_layer

    @staticmethod
    def _connected_components(supra):
        """Yields edge lists of each connected component of a boolean adjacency matrix (upper triangle only)."""
        graph = nx.from_numpy_array(np.triu(supra, k=1))
        for nodes in nx.connected_components(graph):
            if len(nodes) < 2:
                continue
            edges = list(graph.subgraph(nodes).edges())
            if edges:
                yield edges

    @staticmethod
    def calculate_p_values_edges_nbs(true_stability, permutation_stability,
                                     threshold=0.5, component_stat="extent",
                                     alpha=0.05, return_diagnostics=False):
        """Network-Based Statistic (Zalesky et al., 2010) for edge-stability significance.

        Edges whose stability meets ``threshold`` form a graph; its connected
        components are the candidate subnetworks. A permutation null of the
        **largest component statistic** controls the FWER, so a component is
        significant if larger/stronger than the biggest seen in (almost) any
        permutation; its p-value is broadcast onto its member edges (p=1
        elsewhere). Inference is at the *subnetwork* level, not per-edge.

        Stability over K outer folds is discrete ({0, 1/K, ..., 1}), so
        ``threshold=0.5`` keeps edges selected in a majority of folds and
        thresholding is coarse for few folds.

        Args:
            component_stat: ``'extent'`` = edge count (classic NBS);
                ``'intensity'`` = sum of ``(stability - threshold)`` over its edges.
        """
        true_stability = np.asarray(true_stability, dtype=float)
        permutation_stability = np.asarray(permutation_stability, dtype=float)
        n_nodes = true_stability.shape[0]
        n_perms = permutation_stability.shape[-1]
        sig = np.ones((n_nodes, n_nodes, 2))

        if component_stat not in ("extent", "intensity"):
            raise ValueError(
                f"Unknown component_stat '{component_stat}'. Use 'extent' or 'intensity'.")

        def components_with_stats(mat):
            out = []
            for edges in PermutationManager._connected_components(mat >= threshold):
                if component_stat == "extent":
                    stat = float(len(edges))
                else:  # intensity
                    stat = float(sum(mat[i, j] - threshold for i, j in edges))
                out.append((edges, stat))
            return out

        meta = {
            "method": "nbs",
            "threshold": float(threshold),
            "component_stat": component_stat,
            "n_permutations": int(n_perms),
            "alpha": float(alpha),
            "statistic_label": ("Component size (edges)" if component_stat == "extent"
                                else "Component intensity"),
            "networks": {},
        }

        for layer in (Networks.positive, Networks.negative):
            true_layer, perm_layer = PermutationManager._layer_arrays(
                true_stability, permutation_stability, layer)

            max_null = np.zeros(n_perms)
            for p in range(n_perms):
                comps = components_with_stats(perm_layer[:, :, p])
                max_null[p] = max((s for _, s in comps), default=0.0)

            components = []
            for edges, stat in components_with_stats(true_layer):
                p_value = (np.sum(max_null >= stat) + 1) / (n_perms + 1)
                nodes = set()
                for i, j in edges:
                    nodes.update((i, j))
                    sig[i, j, layer] = p_value
                    sig[j, i, layer] = p_value
                components.append({
                    "n_edges": int(len(edges)),
                    "n_nodes": int(len(nodes)),
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "significant": bool(p_value < alpha),
                })

            components.sort(key=lambda c: c["statistic"], reverse=True)
            name = "positive" if layer == Networks.positive else "negative"
            meta["networks"][name] = {
                "max_null": [float(v) for v in max_null],
                "critical_value": float(np.quantile(max_null, 1.0 - alpha)) if n_perms else 0.0,
                "components": components,
                "largest_component_edges": max((c["n_edges"] for c in components), default=0),
                "n_significant_components": int(sum(c["significant"] for c in components)),
            }

        if return_diagnostics:
            return sig, meta
        return sig

    @staticmethod
    def _tfce_map(layer_matrix, heights, E, H, dh):
        """Integrates extent(component)^E * h^H * dh over the threshold sweep `heights`, per edge."""
        n_nodes = layer_matrix.shape[0]
        tfce = np.zeros((n_nodes, n_nodes))
        # np.arange's rounding can put a gridpoint a hair above the intended
        # value (e.g. 0.8 -> 0.8000000000000001); without this tolerance an
        # edge at that value would non-deterministically drop its top contribution.
        tol = dh * 1e-6
        for h in heights:
            for edges in PermutationManager._connected_components(layer_matrix >= h - tol):
                contrib = (len(edges) ** E) * (h ** H) * dh
                for i, j in edges:
                    tfce[i, j] += contrib
                    tfce[j, i] += contrib
        return tfce

    @staticmethod
    def calculate_p_values_edges_tfce(true_stability, permutation_stability,
                                      E=0.5, H=2.0, dh=0.1, alpha=0.05,
                                      return_diagnostics=False):
        """Threshold-Free Cluster Enhancement (Smith & Nichols, 2009) adapted to networks.

        Each edge's TFCE score integrates the support of the components it
        belongs to across a sweep of stability thresholds; a permutation
        **max-TFCE** null across edges gives genuine per-edge FWER-corrected
        p-values (unlike NBS, which is subnetwork-level). E/H are the
        field-standard extent/height exponents.
        """
        true_stability = np.asarray(true_stability, dtype=float)
        permutation_stability = np.asarray(permutation_stability, dtype=float)
        n_nodes = true_stability.shape[0]
        n_perms = permutation_stability.shape[-1]
        heights = np.arange(dh, 1.0 + dh / 2, dh)
        triu = np.triu_indices(n_nodes, k=1)
        sig = np.ones((n_nodes, n_nodes, 2))

        meta = {
            "method": "tfce",
            "E": float(E), "H": float(H), "dh": float(dh),
            "n_permutations": int(n_perms),
            "alpha": float(alpha),
            "statistic_label": "Max TFCE score",
            "networks": {},
        }

        for layer in (Networks.positive, Networks.negative):
            true_layer, perm_layer = PermutationManager._layer_arrays(
                true_stability, permutation_stability, layer)

            true_tfce = PermutationManager._tfce_map(true_layer, heights, E, H, dh)
            max_null = np.zeros(n_perms)
            for p in range(n_perms):
                perm_tfce = PermutationManager._tfce_map(perm_layer[:, :, p], heights, E, H, dh)
                max_null[p] = perm_tfce[triu].max(initial=0.0)

            for i, j in zip(*triu):
                score = true_tfce[i, j]
                if score > 0:
                    p_value = (np.sum(max_null >= score) + 1) / (n_perms + 1)
                    sig[i, j, layer] = p_value
                    sig[j, i, layer] = p_value

            name = "positive" if layer == Networks.positive else "negative"
            meta["networks"][name] = {
                "max_null": [float(v) for v in max_null],
                "critical_value": float(np.quantile(max_null, 1.0 - alpha)) if n_perms else 0.0,
                "observed_max": float(true_tfce[triu].max(initial=0.0)),
                "n_significant_edges": int(np.sum(sig[:, :, layer][triu] < alpha)),
            }

        if return_diagnostics:
            return sig, meta
        return sig
