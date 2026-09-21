"""
Permutation inference.

Everything that turns a null distribution into a p-value: the model-level
permutation test over CV metrics, and the two cluster-based corrections for the
edge-level stability map -- NBS (network-based statistic: suprathreshold
connected components) and TFCE (threshold-free cluster enhancement).

This is the most statistically delicate code in the package. It used to sit in
`results_manager.py` behind ~375 lines of unrelated accumulator, which is a bad
place for it to be hard to find.

The permutation *results* are produced upstream -- `CPMAnalysis` runs the model
on shuffled targets, vectorised over the runs axis. This module only reads the
true-vs-null arrays back off disk and does the statistics.
"""
import os
import json

import numpy as np
import pandas as pd

import networkx as nx

from cccpm.constants import Networks
from cccpm.results_manager import ResultsManager


class PermutationManager:
    @staticmethod
    def calculate_p_values(true_results, perms):
        """
        Calculate p-values based on true results and permutation results.

        :param true_results: DataFrame with the true results.
        :param perms: DataFrame with the permutation results.
        :return: DataFrame with the calculated p-values.
        """
        grouped_true = true_results.groupby(['network', 'model'])
        grouped_perms = perms.groupby(['network', 'model'])

        p_values = []
        for (name, true_group), (_, perms_group) in zip(grouped_true, grouped_perms):
            p_value_series = PermutationManager._calculate_group_p_value(true_group, perms_group)
            p_values.append(pd.DataFrame(p_value_series).T.assign(network=name[0], model=name[1]))

        p_values_df = pd.concat(p_values).reset_index(drop=True)
        p_values_df = p_values_df.set_index(['network', 'model'])
        return p_values_df

    # Metrics where lower is better (p-value: true < perm)
    LOWER_IS_BETTER = {'mean_squared_error', 'mean_absolute_error'}

    @staticmethod
    def _is_lower_better(column_name):
        """Check if a metric is one where lower values are better."""
        if column_name in PermutationManager.LOWER_IS_BETTER:
            return True
        if column_name.endswith('error'):
            return True
        return False

    @staticmethod
    def _calculate_group_p_value(true_group, perms_group):
        """
        Calculate p-value for a group of metrics.

        :param true_group: DataFrame with the true results.
        :param perms_group: DataFrame with the permutation results.
        :return: Series with calculated p-values.
        """
        result_dict = {}
        for column in true_group.columns:
            observed = true_group[column].values[0]
            null = perms_group[column].astype(float)

            # An undefined statistic has no null to compare against. Every
            # comparison with NaN is False, so the count would come out 0 and
            # the +1 correction would report the *floor* -- 1/(n_perms+1), the
            # most significant p-value the test can produce -- for a model that
            # does not exist. (This is what a run without covariates hits: the
            # covariates/full/residuals/increment rows are NaN placeholders.)
            if pd.isna(observed) or null.isna().all():
                result_dict[column] = float('nan')
                continue

            if PermutationManager._is_lower_better(column):
                condition_count = (observed > null).sum()
            else:
                # Higher is better: score, accuracy, balanced_accuracy, f1_score, roc_auc, etc.
                condition_count = (observed < null).sum()

            # Standard permutation p-value (Phipson & Smyth, 2010): the +1 in both
            # numerator and denominator counts the observed statistic itself and
            # guarantees a valid p-value in (0, 1].
            result_dict[column] = (condition_count + 1) / (len(null) + 1)

        return pd.Series(result_dict)

    @staticmethod
    def calculate_permutation_results(results_directory, logger, method="nbs",
                                      nbs_threshold=0.5, nbs_component_stat="extent"):
        """
        Calculate and save the permutation test results.

        Model-level metric p-values are always computed. Edge-stability
        significance is established at the *subnetwork* level via a
        Network-Based Statistic (``method='nbs'``) or, threshold-free, via
        network TFCE (``method='tfce'``); both control the family-wise error
        rate through a permutation max-statistic and write a per-edge p-value
        matrix (edges belonging to a significant subnetwork carry that
        subnetwork's p-value) to ``stability_edges_significance.npy``.

        :param results_directory: Directory where the results are saved.
        :param logger: Logger for progress messages.
        :param method: Edge-significance method, ``'nbs'`` (default) or ``'tfce'``.
        :param nbs_threshold: Stability threshold for NBS component forming.
        :param nbs_component_stat: NBS component statistic, ``'extent'`` or ``'intensity'``.
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
                threshold=nbs_threshold, component_stat=nbs_component_stat,
                return_diagnostics=True)
        elif method == "tfce":
            stability_significance, sig_meta = PermutationManager.calculate_p_values_edges_tfce(
                true_edge_stability, perm_edge_stability, return_diagnostics=True)
        else:
            raise ValueError(
                f"Unknown edge-significance method '{method}'. Use 'nbs' or 'tfce'.")

        np.save(os.path.join(results_directory, 'stability_edges_significance.npy'), stability_significance)
        with open(os.path.join(results_directory, 'stability_edges_significance_meta.json'), 'w') as f:
            json.dump(sig_meta, f)

        logger.debug("Saving significance of edge stability.")
        logger.info("Permutation test results")
        logger.info(p_values.round(4).to_string())
        return

    @staticmethod
    def _layer_arrays(true_stability, permutation_stability, layer):
        """Extract the observed ``[n_nodes, n_nodes]`` matrix and the
        ``[n_nodes, n_nodes, n_perms]`` null stack for one network *layer*
        (``0`` = positive, ``1`` = negative) from the stored stability arrays,
        which have shape ``[n_nodes, n_nodes, 2, runs]``."""
        true_layer = true_stability[:, :, layer, 0]
        perm_layer = permutation_stability[:, :, layer, :]
        return true_layer, perm_layer

    @staticmethod
    def _connected_components(supra):
        """Yield the edge lists of the connected components (each with at least
        one edge) of a symmetric boolean adjacency matrix, using its upper
        triangle only. Isolated nodes are skipped."""
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
        """
        Network-Based Statistic (Zalesky et al., 2010) for edge-stability
        significance.

        Edges whose stability meets ``threshold`` form a graph; its connected
        components are the candidate subnetworks. A permutation null of the
        **largest component statistic** controls the family-wise error rate, so
        an observed component is significant if it is larger/stronger than the
        biggest component seen in (almost) any permutation. Each component's
        p-value is broadcast onto all of its member edges; every other edge is
        assigned ``p = 1``.

        Inference is at the *subnetwork* level: a significant result licenses
        "this connected subnetwork is selected more consistently than chance",
        not per-edge claims.

        Note on discreteness: stability over ``K`` outer folds takes only the
        values ``{0, 1/K, ..., 1}``, so ``threshold=0.5`` keeps edges selected
        in a majority of folds and the effective thresholding is coarse for few
        folds. A continuous edge statistic (deferred) would sharpen this.

        Parameters
        ----------
        true_stability : ndarray of shape (n_nodes, n_nodes, 2, 1)
            Observed edge stability; dim 2 is the positive/negative network.
        permutation_stability : ndarray of shape (n_nodes, n_nodes, 2, n_perms)
            Edge stability from each permutation run.
        threshold : float, default=0.5
            Stability threshold (``>=``) for component forming.
        component_stat : {'extent', 'intensity'}, default='extent'
            ``'extent'`` = number of edges in the component (classic NBS);
            ``'intensity'`` = sum of ``(stability - threshold)`` over its edges.
        alpha : float, default=0.05
            Significance level recorded in the diagnostics.
        return_diagnostics : bool, default=False
            If ``True``, also return a JSON-serialisable diagnostics dict with
            the per-network max-component null distribution, the observed
            components (size / statistic / p-value) and the largest component.

        Returns
        -------
        sig_stability : ndarray of shape (n_nodes, n_nodes, 2)
            Per-edge p-values (member edges carry their component's p-value).
        diagnostics : dict, optional
            Returned only when ``return_diagnostics=True``.
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

            # Null distribution of the largest component statistic.
            max_null = np.zeros(n_perms)
            for p in range(n_perms):
                comps = components_with_stats(perm_layer[:, :, p])
                max_null[p] = max((s for _, s in comps), default=0.0)

            # Observed components -> component p-value on every member edge.
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
        """Threshold-Free Cluster Enhancement score per edge for one network
        layer: integrate ``extent(component)^E * h^H * dh`` over the threshold
        sweep *heights* (extent = number of edges in the component the edge
        belongs to at height ``h``)."""
        n_nodes = layer_matrix.shape[0]
        tfce = np.zeros((n_nodes, n_nodes))
        # Tolerance so an edge whose stability lands exactly on a sweep height is
        # reliably included there. ``heights`` comes from ``np.arange``, whose
        # accumulated rounding can place a gridpoint a hair above the intended
        # value (e.g. 0.8 -> 0.8000000000000001); without this, an edge at that
        # value would non-deterministically drop its top contribution.
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
        """
        Threshold-Free Cluster Enhancement (Smith & Nichols, 2009) adapted to
        networks, for per-edge stability significance without an arbitrary
        primary threshold.

        Each edge's TFCE score integrates the support of the components it
        belongs to across a sweep of stability thresholds. A permutation
        **max-TFCE** null across edges controls the family-wise error rate, so
        this yields genuine per-edge FWER-corrected p-values (unlike NBS, which
        is subnetwork-level).

        Parameters
        ----------
        true_stability : ndarray of shape (n_nodes, n_nodes, 2, 1)
            Observed edge stability; dim 2 is the positive/negative network.
        permutation_stability : ndarray of shape (n_nodes, n_nodes, 2, n_perms)
            Edge stability from each permutation run.
        E, H : float
            TFCE extent/height exponents (field-standard defaults 0.5 / 2.0).
        dh : float, default=0.1
            Step of the stability-threshold sweep over ``(0, 1]``.
        alpha : float, default=0.05
            Significance level recorded in the diagnostics.
        return_diagnostics : bool, default=False
            If ``True``, also return a JSON-serialisable diagnostics dict with
            the per-network max-TFCE null distribution and observed maximum.

        Returns
        -------
        sig_stability : ndarray of shape (n_nodes, n_nodes, 2)
            Per-edge FWER-corrected p-values.
        diagnostics : dict, optional
            Returned only when ``return_diagnostics=True``.
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
