"""
Edge selection: which edges enter the model.

The statistics themselves live in `statistics.py`. This module is the policy
layer on top of them -- the significance thresholding (`PThreshold`), the two
structural filters (presence, connected components), and the
`UnivariateEdgeSelection` facade that expands a user's configuration into the
parameter grid the inner CV iterates over.
"""
import warnings
from typing import Union

import numpy as np

import networkx as nx
import torch

from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid

from cccpm.statistics import correlations_and_pvalues, torch_bonferroni
from cccpm.validation import infer_n_nodes


def resolve_presence_threshold(presence_filter):
    """
    Resolve the ``presence_filter`` argument to a nonzero-fraction threshold.

    Returns ``None`` when the filter is off, otherwise the fraction of subjects
    that must have a nonzero value for an edge to be kept (``True`` -> ``0.5``).
    """
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
    """
    Resolve the ``connected_components`` argument to a minimum component size
    (in edges), or ``None`` when the filter is off (``True`` -> ``2``, i.e. drop
    lone single edges).
    """
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
    """
    Keep only selected edges that belong to a connected component with at least
    ``min_edges`` edges; drop the rest.

    A graph is built independently for each (network, *batch) slice from that
    slice's selected edges, treating edges sharing a node as connected. Isolated
    single edges form a one-edge component and are removed when
    ``min_edges >= 2``.

    ``mask`` is ``[Features, 2, *batch]`` (dim 1 = positive/negative; batch is
    typically params x runs). The returned tensor has the same shape and dtype,
    with dropped edges set to 0.

    This is CPU/networkx work and is not vectorised -- it runs once per
    (network, *batch) slice.
    """
    n_features = mask.shape[0]
    n_nodes = infer_n_nodes(n_features)
    if n_nodes is None:
        return mask

    rows, cols = np.triu_indices(n_nodes, k=1)
    out = mask.clone()
    selected = mask.detach().cpu().numpy() > 0

    # Flatten every axis after [Features, 2] so one loop covers any batch shape.
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
    def select(self, r, p, thresholds=None):
        pass


class PThreshold(BaseEdgeSelector):
    """
    Select edges whose correlation with the target is significant at a p-value
    threshold, optionally after multiple-comparison correction.

    Pass a list of thresholds (and/or corrections) to search over them with an
    inner cross-validation loop.
    """
    def __init__(self, threshold: Union[float, list] = 0.05, correction: Union[str, list] = None):
        """
        :param threshold: p-value threshold(s); edges with p below the threshold
                          are selected. A single value (e.g. ``0.05``) or a list
                          (e.g. ``[0.01, 0.05]``) to tune via inner CV.
        :param correction: multiple-comparison correction, or ``None`` for no
                            correction. Can be one of statsmodels' methods:
                            bonferroni : one-step correction
                            sidak : one-step correction
                            holm-sidak : step down method using Sidak adjustments
                            holm : step-down method using Bonferroni adjustments
                            simes-hochberg : step-up method (independent)
                            hommel : closed method based on Simes tests (non-negative)
                            fdr_bh : Benjamini/Hochberg (non-negative)
                            fdr_by : Benjamini/Yekutieli (negative)
                            fdr_tsbh : two stage fdr correction (non-negative)
                            fdr_tsbky : two stage fdr correction (non-negative)
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

    def _apply_correction(self, p):
        """Multiple-comparison-correct `p`, returning a tensor of the same shape."""
        if self._correction is None:
            return p
        if self._correction == 'bonferroni':
            # Stays on-device (no CPU/GPU sync), unlike the statsmodels path below.
            _, p_corrected = torch_bonferroni(p)
            return p_corrected
        # Other statsmodels corrections still require a CPU round-trip.
        from statsmodels.stats import multitest
        shape = p.shape
        p_np = p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else p
        _, p_flat, _, _ = multitest.multipletests(p_np.flatten(), alpha=0.05, method=self._correction)
        p_flat = p_flat.reshape(shape)
        return torch.as_tensor(p_flat, device=p.device, dtype=p.dtype) if isinstance(p, torch.Tensor) else p_flat

    def select(self, r, p, thresholds=None):
        """
        Select edges whose (optionally corrected) p-value falls below each
        threshold, split into positive- and negative-correlation networks.

        The correction is applied once and then compared against every
        threshold at the same time, so searching a threshold grid costs one
        call rather than one call per value. All thresholds share this
        selector's single `correction` method -- different correction methods
        use different formulas and need separate calls.

        Args:
            r, p: [N_features, *rest], `rest` being any batch axes already
                  present (in practice the runs/permutations axis).
            thresholds: sequence of p-value thresholds. Defaults to
                        ``self.threshold``, which is always a list.

        Returns:
            Boolean tensor [N_features, 2, N_params, *rest], where dim 1 is
            [positive, negative] and N_params == len(thresholds).
        """
        if thresholds is None:
            thresholds = self.threshold

        p_corrected = self._apply_correction(p)
        rest_shape = p_corrected.shape[1:]
        thresholds_t = torch.as_tensor(list(thresholds), device=r.device,
                                       dtype=p_corrected.dtype)
        thresh_view = thresholds_t.view(1, -1, *([1] * len(rest_shape)))

        p_exp = p_corrected.unsqueeze(1)   # [N_features, 1, *rest]
        r_exp = r.unsqueeze(1)             # [N_features, 1, *rest]
        pos_mask = (p_exp < thresh_view) & (r_exp > 0)
        neg_mask = (p_exp < thresh_view) & (r_exp < 0)

        return torch.stack([pos_mask, neg_mask], dim=1)


SELECTION_STATISTICS = ('pearson', 'spearman')
SELECTION_INPUTS = ('raw', 'residualized')

# Legacy ``edge_statistic`` value -> (selection_statistic, selection_input).
#
# The six old values were really a 2x2 (plus two aliases): which correlation,
# and whether the confounds are controlled for. ``point_biserial`` was never a
# separate statistic -- it is Pearson against a 0/1 target, which the unified
# OLS path already handles with no special-casing.
_LEGACY_EDGE_STATISTICS = {
    'pearson': ('pearson', 'raw'),
    'spearman': ('spearman', 'raw'),
    'point_biserial': ('pearson', 'raw'),
    'pearson_partial': ('pearson', 'residualized'),
    'spearman_partial': ('spearman', 'residualized'),
    'point_biserial_partial': ('pearson', 'residualized'),
}


def resolve_selection_spec(selection_statistic, selection_input, edge_statistic):
    """
    Resolve the edge-selection specification, honouring the deprecated
    ``edge_statistic`` argument for one release.

    Returns ``(selection_statistic, selection_input)``.
    """
    if edge_statistic is not None:
        if edge_statistic not in _LEGACY_EDGE_STATISTICS:
            raise ValueError(
                f"Unknown edge_statistic {edge_statistic!r}. Valid values were "
                f"{sorted(_LEGACY_EDGE_STATISTICS)}; this parameter is deprecated, "
                f"use selection_statistic and selection_input instead."
            )
        statistic, selection = _LEGACY_EDGE_STATISTICS[edge_statistic]
        warnings.warn(
            f"edge_statistic={edge_statistic!r} is deprecated and will be removed "
            f"in a future release. Use selection_statistic={statistic!r}, "
            f"selection_input={selection!r} instead.",
            DeprecationWarning, stacklevel=3,
        )
        return statistic, selection

    if selection_statistic not in SELECTION_STATISTICS:
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
    """
    The per-edge statistic used to rank and threshold edges.

    Two independent choices:

    ``selection_statistic``
        ``'pearson'`` or ``'spearman'``. A binary 0/1 target through the Pearson
        path *is* the point-biserial correlation, so it needs no separate value.

    ``selection_input``
        ``'raw'`` ignores the covariates. ``'residualized'`` controls for them:
        one regression per edge, ``y ~ 1 + Z + edge``, reporting the semipartial
        correlation as the effect size and the coefficient's p-value with
        ``df = N - 2 - C``.

    On ``'residualized'``: the effect size is reported as the *semipartial*
    correlation -- the edge's unique contribution as a share of the total
    variance of y -- because that denominator stays comparable across analyses.
    The partial correlation is the same effect on a denominator that shrinks
    with how confounded y is. They are monotone transforms of each other within
    a fold (verified: rank correlation exactly 1.0), so the choice affects what
    is printed, never which edges are selected. Both come from one regression,
    which is where the p-value comes from too.

    For Spearman the ranking happens *first*, including on the confounds, and
    the residualisation follows -- the conventional "rank, then partial"
    definition. The reverse order does not work: ranking is nonlinear, so
    ranking a residualised edge puts the confound signal back in (measured:
    11.9 versus 6.5e-13 of residual confound signal).
    """

    def __init__(self, selection_statistic: str = 'spearman',
                 selection_input: str = 'raw',
                 presence_filter: Union[bool, float] = False,
                 edge_statistic: str = None):
        # Stored verbatim for sklearn's get_params contract; the resolved pair
        # lives in the private attributes below.
        self.selection_statistic = selection_statistic
        self.selection_input = selection_input
        self.presence_filter = presence_filter
        self.edge_statistic = edge_statistic
        self._statistic, self._input = resolve_selection_spec(
            selection_statistic, selection_input, edge_statistic)

    def fit_transform(self,
                      X,
                      y,
                      covariates,
                      device):
        r_edges, p_edges = (torch.zeros((X.shape[1], y.shape[1]), device=device),
                            torch.ones((X.shape[1], y.shape[1]), device=device))

        # 1. Convert to GPU Tensors immediately
        X = torch.as_tensor(X, device=device, dtype=torch.float32)
        y = torch.as_tensor(y, device=device, dtype=torch.float32)
        if covariates is not None:
            covariates = torch.as_tensor(covariates, device=device, dtype=torch.float32)

        # 3. Variance Threshold (GPU Version)
        # Replaces sklearn.feature_selection.VarianceThreshold
        # Remove features with ~0 variance to avoid NaNs in correlation
        variances = torch.var(X, dim=0)
        valid_edges = variances > 1e-6

        # 3b. Presence filter (optional): drop edges that are zero for more than
        # (1 - threshold) of the subjects. Intended for sparse structural
        # connectomes (e.g. DTI streamline counts) where structural zeros should
        # not enter the model; leave off for functional data whose edges have a
        # real signed distribution around a mean of ~0. Uses X only (no target),
        # computed here on the training subjects, so it adds no leakage. This is
        # additive to the variance gate above, which already drops all-zero edges.
        # It always sees the raw connectome: confound control happens inside the
        # statistic below, never by residualising X before this point.
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
        r_edges_masked, p_edges_masked = correlations_and_pvalues(
            X=X, Y_perms=y, confounds=confounds,
            correlation_type=self._statistic)

        # no dynamic shape change
        mask = valid_edges.to(r_edges_masked.dtype).unsqueeze(1)
        r_edges = r_edges_masked.to(r_edges.dtype) * mask
        p_edges = p_edges_masked.to(p_edges.dtype) * mask + (1.0 - mask)
        return r_edges, p_edges


class UnivariateEdgeSelection(BaseEstimator):
    """
    Univariate edge selection for CPM.

    Correlates each edge with the target using the chosen statistic and selects
    edges with one or more selection strategies (e.g. a p-value threshold). When
    several configurations are supplied, they form a hyperparameter grid that the
    inner cross-validation loop searches over.

    Parameters
    ----------
    selection_statistic: str, default='spearman'
        Correlation used to relate each edge to the target: ``'pearson'`` or
        ``'spearman'``. A binary 0/1 target through the Pearson path is the
        point-biserial correlation, so it needs no separate value.
    selection_input: str, default='raw'
        Whether edge selection controls for the covariates. ``'raw'`` ignores
        them. ``'residualized'`` fits one regression per edge,
        ``y ~ 1 + Z + edge``, reporting the semipartial correlation and the
        coefficient's p-value (``df = N - 2 - C``) -- so a ``p < 0.05``
        threshold means a 5% per-edge false-positive rate whatever the
        confounding. It requires covariates.
    edge_statistic: str, default=None
        .. deprecated::
            Use ``selection_statistic`` and ``selection_input``. The old values
            map as: ``'pearson'``/``'spearman'`` -> ``selection_input='raw'``;
            ``'pearson_partial'``/``'spearman_partial'`` ->
            ``selection_input='residualized'``; ``'point_biserial'`` ->
            ``'pearson'`` with ``'raw'`` (and ``'point_biserial_partial'`` ->
            ``'pearson'`` with ``'residualized'``). This parameter will be
            removed in a future release.
    presence_filter: bool or float, default=False
        Optional pre-filter that keeps only edges which are nonzero in at least a
        given fraction of subjects, dropping structural/near-zero edges before
        selection. ``True`` uses a fraction of ``0.5`` (present in the majority);
        a float sets the fraction explicitly (e.g. ``0.75``). Intended for sparse
        structural connectomes (e.g. DTI streamline counts); leave off
        (``False``) for functional data, whose edges have a real signed
        distribution around a mean of ~0. Computed per fold on the training
        subjects from the connectome only, so it adds no target leakage. It
        always sees the raw connectome, including under
        ``selection_input='residualized'`` -- confound control now happens
        inside the statistic rather than by residualising X beforehand.
    connected_components: bool or int, default=False
        If set, keep only selected edges that belong to a connected component
        with at least this many edges (per positive/negative network), dropping
        isolated edges — this can improve edge stability. ``True`` uses a minimum
        of ``2`` edges (drop lone single edges); an int sets the minimum
        explicitly. Applied per fold and per permutation after thresholding.
    edge_selection: list of selectors (e.g. PThreshold), default=None
        One or more selection strategies. Provide a list of multiple
        configurations to tune them via an inner CV.
    """
    def __init__(self,
                 selection_statistic: str = 'spearman',
                 selection_input: str = 'raw',
                 presence_filter: Union[bool, float] = False,
                 connected_components: Union[bool, int] = False,
                 edge_selection: Union[list, None, PThreshold] = None,
                 edge_statistic: str = None):
        self.r_edges = None
        self.p_edges = None
        self.selection_statistic = selection_statistic
        self.selection_input = selection_input
        self.presence_filter = presence_filter
        self.connected_components = connected_components
        self.edge_statistic = edge_statistic
        self.statistic = EdgeStatistic(selection_statistic=selection_statistic,
                                       selection_input=selection_input,
                                       presence_filter=presence_filter,
                                       edge_statistic=edge_statistic)
        self.edge_selection = edge_selection
        if isinstance(edge_selection, (list, tuple)):
            self.edge_selection = edge_selection
        else:
            self.edge_selection = [edge_selection]
        self.param_grid = self._generate_config_grid()

    def _generate_config_grid(self):
        grid_elements = []
        for selector in self.edge_selection:
            params = {}
            params['edge_selection'] = [selector]
            for key, value in selector.get_params().items():
                params['edge_selection__' + key] = value
            grid_elements.append(params)
        return ParameterGrid(grid_elements)

    def fit_transform(self, X, y=None, covariates=None, device=torch.device('cpu')):
        self.r_edges, self.p_edges = self.statistic.fit_transform(X=X, y=y, covariates=covariates, device=device)
        return self

    def return_selected_edges(self, thresholds=None):
        """Edge masks [Features, 2, N_params, *runs] for the fitted r/p values."""
        selected_edges = self.edge_selection.select(
            r=self.r_edges, p=self.p_edges, thresholds=thresholds)
        min_edges = resolve_min_component_size(self.connected_components)
        if min_edges is not None:
            selected_edges = filter_connected_components(selected_edges, min_edges)
        return selected_edges
