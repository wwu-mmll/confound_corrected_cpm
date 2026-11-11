"""Policy layer over `statistics.py`: p-value thresholding, structural filters, and the parameter grid."""
from typing import Union

import numpy as np

import networkx as nx
import torch

from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid

from cccpm.statistics import edge_statistics, t_pvalues, critical_t
from cccpm.validation import infer_n_nodes


def resolve_presence_threshold(presence_filter):
    """Resolves ``presence_filter`` to a nonzero-fraction threshold (``True`` -> ``0.5``), or ``None`` if off."""
    if presence_filter is False or presence_filter is None:
        return None
    if presence_filter is True:
        return 0.5
    threshold = float(presence_filter)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"presence_filter must be a bool or a fraction in [0, 1], "
            f"got {presence_filter!r}."
        )
    return threshold


def resolve_min_component_size(connected_components):
    """Resolves ``connected_components`` to a minimum component size in edges (``True`` -> ``2``), or ``None`` if off."""
    if connected_components is False or connected_components is None:
        return None
    if connected_components is True:
        return 2
    size = int(connected_components)
    if size < 1:
        raise ValueError(
            f"connected_components must be a bool or an int >= 1, "
            f"got {connected_components!r}."
        )
    return size


def filter_connected_components(mask, min_edges):
    """Zeroes out selected edges not in a component with >= min_edges edges.

    Built per (network, *batch) slice with networkx (CPU, not vectorised).
    `mask` shape is [Features, 2, *batch] (dim 1 = positive/negative).
    """
    n_features = mask.shape[0]
    n_nodes = infer_n_nodes(n_features)
    if n_nodes is None:
        return mask

    rows, cols = np.triu_indices(n_nodes, k=1)
    out = mask.clone()
    selected = mask.detach().cpu().numpy() > 0

    # Flatten axes after [Features, 2] so one loop covers any batch shape.
    n_layers = selected.shape[1]
    batch_shape = selected.shape[2:]
    flat = selected.reshape(n_features, n_layers, -1)
    out_flat = out.reshape(n_features, n_layers, -1)

    for layer in range(n_layers):
        for b in range(flat.shape[2]):
            edge_idx = np.nonzero(flat[:, layer, b])[0]
            if edge_idx.size == 0:
                continue
            graph = nx.Graph()
            graph.add_edges_from(zip(rows[edge_idx].tolist(), cols[edge_idx].tolist()))
            drop = set()
            for component in nx.connected_components(graph):
                sub = graph.subgraph(component)
                if sub.number_of_edges() < min_edges:
                    drop.update(tuple(sorted(e)) for e in sub.edges())
            if drop:
                for e in edge_idx:
                    if (rows[e], cols[e]) in drop:
                        out_flat[e, layer, b] = 0

    return out_flat.reshape(n_features, n_layers, *batch_shape)


class BaseEdgeSelector(BaseEstimator):
    """Interface for edge selectors: implement `select` to return a boolean edge mask."""

    def select(self, r, p, thresholds=None):
        pass


class PThreshold(BaseEdgeSelector):
    """Selects edges significant at a p-value threshold, with optional multiple-comparison correction."""
    def __init__(self, threshold: Union[float, list] = 0.05, correction: Union[str, list] = None):
        """Each arg may be a list to tune via inner CV.

        Args:
            correction: ``None``, or a statsmodels method name, e.g.
                bonferroni/sidak/holm/fdr_bh.
        """
        self._threshold = None
        self._correction = None
        self.threshold = threshold
        self.correction = correction

    @property
    def threshold(self):
        if isinstance(self._threshold, (int, float)):
            return [float(self._threshold)]
        return self._threshold or [0.05]

    @threshold.setter
    def threshold(self, value):
        if isinstance(value, (int, float)):
            self._threshold = float(value)
        elif isinstance(value, list):
            self._threshold = value
        else:
            raise ValueError("threshold must be float or list")

    @property
    def correction(self):
        if self._correction is None:
            return [None]
        if isinstance(self._correction, str):
            return [self._correction]
        return self._correction

    @correction.setter
    def correction(self, value):
        if value is None:
            self._correction = None
        elif isinstance(value, str):
            self._correction = value
        elif isinstance(value, list):
            self._correction = value
        else:
            raise ValueError("correction must be None, str, or list")

    def _p_cutoffs(self, thresholds, n_tests):
        """Per-threshold uncorrected p cut-offs for none/bonferroni/sidak (else None); avoids per-edge p-values."""
        if self._correction is None:
            return [float(a) for a in thresholds]
        if self._correction == 'bonferroni':
            return [float(a) / n_tests for a in thresholds]
        if self._correction == 'sidak':
            return [1.0 - (1.0 - float(a)) ** (1.0 / n_tests) for a in thresholds]
        return None

    def _stepwise_significant(self, t, df, thresholds):
        """[N_features, N_thresholds, *rest] mask for step-wise corrections (Holm, FDR, ...), run one at a time."""
        from statsmodels.stats import multitest
        p = t_pvalues(t, df).cpu().double().numpy()
        n_features, rest_shape = p.shape[0], p.shape[1:]
        columns = p.reshape(n_features, -1)
        corrected = np.empty_like(columns)
        for run in range(columns.shape[1]):
            _, corrected[:, run], _, _ = multitest.multipletests(
                columns[:, run], alpha=0.05, method=self._correction)
        corrected = corrected.reshape(n_features, 1, *rest_shape)
        thresh = np.asarray(thresholds, dtype=np.float64).reshape(1, -1, *([1] * len(rest_shape)))
        return torch.as_tensor(corrected < thresh, device=t.device)

    def select(self, r, t, df, thresholds=None):
        """Selects edges below each p-value threshold, split into [positive, negative] networks (dim 1 of the result).

        Exact without a per-edge p-value: for fixed df, p is strictly
        decreasing in |t|, so ``p < alpha`` is ``|t| > critical_t(alpha, df)``.
        Correction is applied per run (per column of `t`) -- a permuted run
        must be thresholded exactly like the real one, or the null isn't a
        null of the same procedure. Returns [N_features, 2, N_params, *rest].
        """
        if thresholds is None:
            thresholds = self.threshold
        rest_shape = t.shape[1:]

        cutoffs = self._p_cutoffs(thresholds, n_tests=t.shape[0])
        if cutoffs is not None:
            crit = torch.as_tensor([critical_t(a, df) for a in cutoffs],
                                   device=t.device, dtype=t.dtype)
            crit = crit.view(1, -1, *([1] * len(rest_shape)))
            significant = t.abs().unsqueeze(1) > crit          # [F, N_params, *rest]
        else:
            significant = self._stepwise_significant(t, df, thresholds)

        r_exp = r.unsqueeze(1)                                 # [F, 1, *rest]
        pos_mask = significant & (r_exp > 0)
        neg_mask = significant & (r_exp < 0)
        return torch.stack([pos_mask, neg_mask], dim=1)


SELECTION_STATISTICS = ('pearson', 'spearman')
SELECTION_INPUTS = ('raw', 'residualized')

# Removed edge_statistic values -> their (statistic, input) replacement, for the error below.
_REMOVED_EDGE_STATISTICS = {
    'pearson': ('pearson', 'raw'),
    'spearman': ('spearman', 'raw'),
    'point_biserial': ('pearson', 'raw'),
    'pearson_partial': ('pearson', 'residualized'),
    'spearman_partial': ('spearman', 'residualized'),
    'point_biserial_partial': ('pearson', 'residualized'),
}


def resolve_selection_spec(selection_statistic, selection_input):
    """Validates the pair, raising a pointer to the replacement if a removed `edge_statistic` value was passed."""
    if selection_statistic not in SELECTION_STATISTICS:
        if selection_statistic in _REMOVED_EDGE_STATISTICS:
            statistic, selection = _REMOVED_EDGE_STATISTICS[selection_statistic]
            raise ValueError(
                f"{selection_statistic!r} was an `edge_statistic` value, which "
                f"was removed in 0.7.0. Use selection_statistic={statistic!r} "
                f"with selection_input={selection!r} instead."
                + ("" if selection == 'raw' else
                   " Note that confound-controlled selection now uses the "
                   "per-edge coefficient test (df = N - 2 - C) rather than a "
                   "plain correlation against the raw target, so edge sets "
                   "differ from 0.6.x.")
            )
        raise ValueError(
            f"selection_statistic must be one of {SELECTION_STATISTICS}, "
            f"got {selection_statistic!r}."
        )
    if selection_input not in SELECTION_INPUTS:
        raise ValueError(
            f"selection_input must be one of {SELECTION_INPUTS}, "
            f"got {selection_input!r}."
        )
    return selection_statistic, selection_input


class EdgeStatistic(BaseEstimator):
    """The per-edge statistic used to rank and threshold edges.

    ``selection_input='residualized'`` fits one regression per edge
    (``y ~ 1 + Z + edge``, ``df = N - 2 - C``) and reports the *semipartial*
    correlation (not partial): both are monotone transforms of each other
    within a fold (rank correlation exactly 1.0, so edge selection is
    unaffected), but semipartial keeps a denominator comparable across
    analyses. For Spearman, ranking happens before residualising (ranking a
    residualised edge puts confound signal back in: measured 11.9 vs
    6.5e-13 of residual confound signal).
    """

    def __init__(self, selection_statistic: str = 'spearman',
                 selection_input: str = 'raw',
                 presence_filter: Union[bool, float] = False):
        self.selection_statistic = selection_statistic
        self.selection_input = selection_input
        self.presence_filter = presence_filter
        self._statistic, self._input = resolve_selection_spec(
            selection_statistic, selection_input)

    def fit_transform(self,
                      X,
                      y,
                      covariates,
                      device):
        """Returns (effect size, t statistic, df); see `statistics.edge_statistics`. Filtered-out edges get r = t = 0."""
        X = torch.as_tensor(X, device=device, dtype=torch.float32)
        y = torch.as_tensor(y, device=device, dtype=torch.float32)
        if covariates is not None:
            covariates = torch.as_tensor(covariates, device=device, dtype=torch.float32)

        # Drop near-zero-variance edges to avoid NaNs in the correlation.
        variances = torch.var(X, dim=0)
        valid_edges = variances > 1e-6

        # Presence filter sees the raw connectome even under
        # selection_input='residualized' -- confound control happens inside
        # the statistic below, never by residualising X beforehand.
        presence_threshold = resolve_presence_threshold(self.presence_filter)
        if presence_threshold is not None:
            presence = (X != 0).float().mean(dim=0)
            valid_edges = valid_edges & (presence >= presence_threshold)

        confounds = covariates if self._input == 'residualized' else None
        if confounds is None and self._input == 'residualized':
            raise ValueError(
                "selection_input='residualized' requires covariates, but none "
                "were supplied. Use selection_input='raw' for an analysis "
                "without confound control."
            )
        r_edges, t_edges, df = edge_statistics(
            X=X, Y_perms=y, confounds=confounds,
            correlation_type=self._statistic)

        # Excluded edges get r = t = 0, which no threshold selects.
        mask = valid_edges.to(r_edges.dtype).unsqueeze(1)
        return r_edges * mask, t_edges * mask, df


class UnivariateEdgeSelection(BaseEstimator):
    """Univariate edge selection for CPM.

    Correlates each edge with the target using the chosen statistic and selects
    edges with one or more selection strategies (e.g. a p-value threshold). When
    several configurations are supplied, they form a hyperparameter grid that the
    inner cross-validation loop searches over.
    """
    def __init__(self,
                 selection_statistic: str = 'spearman',
                 selection_input: str = 'raw',
                 presence_filter: Union[bool, float] = False,
                 connected_components: Union[bool, int] = False,
                 edge_selection: Union[list, None, PThreshold] = None):
        """Configures edge selection.

        Args:
            selection_input: ``'raw'`` ignores covariates; ``'residualized'``
                fits one regression per edge and requires covariates (see
                `EdgeStatistic`).
            presence_filter: Drop edges nonzero in less than this fraction of
                subjects (``True`` -> ``0.5``); for sparse structural
                connectomes (e.g. DTI counts), not functional data.
            connected_components: Keep only edges in a component with at
                least this many edges per network (``True`` -> ``2``);
                improves edge stability.
            edge_selection: One or more selectors (e.g. `PThreshold`); a list
                tunes them via inner CV.
        """
        self.r_edges = None
        self.t_edges = None
        self.df = None
        self.selection_statistic = selection_statistic
        self.selection_input = selection_input
        self.presence_filter = presence_filter
        self.connected_components = connected_components
        self.statistic = EdgeStatistic(selection_statistic=selection_statistic,
                                       selection_input=selection_input,
                                       presence_filter=presence_filter)
        self.edge_selection = edge_selection
        if isinstance(edge_selection, (list, tuple)):
            self.edge_selection = edge_selection
        else:
            self.edge_selection = [edge_selection]
        self.param_grid = self._generate_config_grid()

    def _generate_config_grid(self):
        """Expands `self.edge_selection` into a sklearn `ParameterGrid` over each selector's own params."""
        grid_elements = []
        for selector in self.edge_selection:
            params = {}
            params['edge_selection'] = [selector]
            for key, value in selector.get_params().items():
                params['edge_selection__' + key] = value
            grid_elements.append(params)
        return ParameterGrid(grid_elements)

    def fit_transform(self, X, y=None, covariates=None, device=torch.device('cpu')):
        """Computes and stores the per-edge effect size and t statistic; returns self."""
        self.r_edges, self.t_edges, self.df = self.statistic.fit_transform(
            X=X, y=y, covariates=covariates, device=device)
        return self

    def return_selected_edges(self, thresholds=None):
        """Edge masks [Features, 2, N_params, *runs] for the fitted statistics."""
        selected_edges = self.edge_selection.select(
            r=self.r_edges, t=self.t_edges, df=self.df, thresholds=thresholds)
        min_edges = resolve_min_component_size(self.connected_components)
        if min_edges is not None:
            selected_edges = filter_connected_components(selected_edges, min_edges)
        return selected_edges
