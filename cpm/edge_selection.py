import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp
from pingouin import partial_corr

from sklearn.base import BaseEstimator

class BaseEdgeSelector(BaseEstimator):
    def transform(self):
        pass


def one_sample_t_test(x):
    # use two-sided for correlations (functional connectome) or one-sided positive for NOS etc (structural connectome)
    _, p_value = ttest_1samp(x, popmean=0, nan_policy='omit', alternative='two-sided')
    return p_value


def partial_correlation(X, y, covariates, method: str = 'pearson'):
    p_values = list()

    df = pd.concat([pd.DataFrame(X, columns=[f"x{s}" for s in range(X.shape[1])]), pd.DataFrame({'y': y})], axis=1)
    cov_names = list()
    for c in range(covariates.shape[1]):
        df[f'cov{c}'] = covariates[:, c]
        cov_names.append(f'cov{c}')

    for xi in range(X.shape[1]):
        res = partial_corr(data=df, x=f'x{xi}', y='y', covar=cov_names, method=method)['p-val'].iloc[0]
        p_values.append(res)
    return np.asarray(p_values)



from typing import Union

class PThreshold(BaseEdgeSelector):
    def __init__(self, threshold: Union[float, list] = 0.05, correction: Union[float, list] = None):
        self.threshold = threshold
        self.correction = correction


class SelectPercentile(BaseEdgeSelector):
    def __init__(self, percentile: Union[float, list] = 0.05):
        self.percentile = percentile


class SelectKBest(BaseEdgeSelector):
    def __init__(self, k: Union[int, list] = None):
        self.k = k

from sklearn.utils.metaestimators import _BaseComposition


class CPMPipeline(_BaseComposition):
    def __init__(self, elements):
        self.elements = elements
        self.current_config = None

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._get_params('elements', deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.
        Valid parameter keys can be listed with ``get_params()``.
        Returns
        -------
        self
        """
        if self.current_config is not None and len(self.current_config) > 0:
            if kwargs is not None and len(kwargs) == 0:
                raise ValueError("Pipeline cannot set parameters to elements with an emtpy dictionary. Old values persist")
        self.current_config = kwargs
        self._set_params('elements', **kwargs)

        return self

    @property
    def named_steps(self):
        return dict(self.elements)


from sklearn.model_selection import ParameterGrid


class UnivariateEdgeSelection(BaseEstimator):
    def __init__(self,
                 edge_statistic: Union[str, list] = 'pearson',
                 edge_selection: list = None):
        self.edge_statistic = edge_statistic
        self.edge_selection = edge_selection
        self.param_grid = self._generate_config_grid()

    def _generate_config_grid(self):
        grid_elements = []
        for selector in self.edge_selection:
            params = {}
            params['edge_statistic'] = self.edge_statistic
            params['edge_selection'] = [selector]
            for key, value in selector.get_params().items():
                params['edge_selection__' + key] = value
            grid_elements.append(params)
        return ParameterGrid(grid_elements)

    def fit_transform(self, X, y=None, covariates=None):
        r_edges, p_edges = self._partial_correlation(X=X, y=y, covariates=covariates)
        pos_edges, neg_edges = self._edge_selection(r=r_edges, p=p_edges, threshold=0.01)
        return pos_edges, neg_edges

    @staticmethod
    def _partial_correlation(X: Union[pd.DataFrame, np.ndarray],
                         y: Union[pd.Series, pd.DataFrame, np.ndarray],
                         covariates: Union[pd.Series, pd.DataFrame, np.ndarray]):
        #p_edges = partial_correlation(X=X, y=y, covariates=covariates)
        r_edges = np.random.randn(X.shape[1])
        p_edges = np.random.randn(X.shape[1])
        return r_edges, p_edges

    @staticmethod
    def _edge_selection(r: np.ndarray,
                        p: np.ndarray,
                        threshold: float):
        pos_edges = np.where((p < threshold) & (r > 0))[0]
        neg_edges = np.where((p < threshold) & (r < 0))[0]
        return pos_edges, neg_edges
