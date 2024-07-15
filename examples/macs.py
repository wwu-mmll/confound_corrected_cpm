import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from cpm import CPMAnalysis


X = np.load('/spm-data/vault-data3/mmll/projects/macs_datahub_example/AnalysisReady/all/FunctionalConnectome/X.npy')
df = pd.read_csv('/spm-data/vault-data3/mmll/projects/macs_datahub_example/AnalysisReady/all/'
                 'FunctionalConnectome/sample.csv',
                 na_values=-99)
X = X[~df['BDI_Sum'].isna()]
df = df[~df['BDI_Sum'].isna()]
covs = df[['Alter', 'Geschlecht']].to_numpy()
y = df['BDI_Sum'].to_numpy()
#covs = df[['Geschlecht']].to_numpy()
#y = df['Alter'].to_numpy()


from sklearn.base import BaseEstimator

class BaseEdgeSelector(BaseEstimator):
    def transform(self):
        pass



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




"""
pipeline.set_params(**{'univariate_edge_selection': select_percentile,
                     'univariate_edge_selection__percentile': 0.5})
"""
p_threshold = PThreshold(threshold=[0.05, 0.01, 0.001], correction=[None, 'FWE', 'FPR', 'FDR'])
select_percentile = SelectPercentile(percentile=[0.5, 0.25])
select_kbest = SelectKBest(k=[5, 10, 15])
univariate_edge_selection = UnivariateEdgeSelection(edge_statistic=['pearson', 'semi-partial'],
                                                    edge_selection=[p_threshold, select_percentile, select_kbest])
from cpm.models import LinearCPMModel

pipeline = CPMPipeline(elements=[('univariate_edge_selection', univariate_edge_selection),
                                 ('linear_model', LinearCPMModel())])


p_threshold.get_params()
"""

cpm = CPMAnalysis(results_directory='./tmp/macs_demo',
                  cv=KFold(n_splits=10, shuffle=True, random_state=42),
                  inner_cv=KFold(n_splits=10, shuffle=True, random_state=42),
                  edge_statistic=edge_statistic,
                  edge_selection=edge_selection,
                  use_covariates=True,
                  estimate_model_increments=True)
results = cpm.fit(X=X, y=y, covariates=covs)
print(results)
p_pos, p_neg = cpm.permutation_test(X=X, y=y, covariates=covs, n_perms=30)

print(p_pos)
print(p_neg)
"""
