from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from pygam import LinearGAM


class NetworkDict(dict):
    def __init__(self):
        super().__init__(self)
        self.update({'positive': {}, 'negative': {}, 'both': {}})

    @staticmethod
    def n_networks():
        return len(NetworkDict().keys())


class ModelDict(dict):
    def __init__(self):
        super().__init__(self)
        self.update({'connectome': {}, 'covariates': {}, 'full': {}, 'residuals': {}})

    @staticmethod
    def n_models():
        return len(ModelDict().keys())


class BaseCPMModel(ABC):
    """
    Base Connectome-based Predictive Modeling (CPM) class.

    Subclasses only need to define:
      - fit_model(self, X, y) -> fitted_estimator
      - predict_model(self, fitted_estimator, X) -> y_pred
    """
    name = 'BaseCPMModel'
    def __init__(self, edges: Dict[str, np.ndarray]):
        """
        Parameters
        ----------
        edges : dict
            Dictionary containing indices of edges for 'positive' and 'negative' networks.
        """
        self.models = ModelDict()
        self.models_residuals: Dict[str, Any] = {}
        self.edges = edges

    # --- Methods to override in subclasses ---
    @abstractmethod
    def fit_model(self, X: np.ndarray, y: np.ndarray):
        """Return a fitted estimator for the provided X, y."""
        raise NotImplementedError

    # --- CPM logic (unchanged across algorithms) ---
    def fit(self, X: np.ndarray, y: np.ndarray, covariates: np.ndarray):
        """
        Fit CPM models for connectome, covariates, residuals, and full variants.
        Residualization of connectome on covariates is kept linear (as in your code).
        """
        connectome = {}
        residuals = {}
        network_strengths = {}

        # Compute network strengths and residualize vs covariates
        for network in ['positive', 'negative']:
            network_strengths[network] = np.sum(X[:, self.edges[network]], axis=1).reshape(-1, 1)
            self.models_residuals[network] = LinearRegression().fit(covariates, network_strengths[network])
            residuals[network] = network_strengths[network] - self.models_residuals[network].predict(covariates)
            if self.edges[network].shape[0] == 0:
                connectome[network] = np.zeros((X.shape[0], 1))
            else:
                connectome[network] = X[:, self.edges[network]]

        # Combine networks
        residuals['both'] = np.hstack((residuals['positive'], residuals['negative']))
        network_strengths['both'] = np.hstack((network_strengths['positive'], network_strengths['negative']))
        if np.concatenate([X[:, self.edges['positive']], X[:, self.edges['negative']]], axis=1).shape[1] == 0:
            connectome['both'] = np.zeros((X.shape[0], 1))
        else:
            connectome['both'] = np.concatenate([X[:, self.edges['positive']], X[:, self.edges['negative']]], axis=1)

        # Fit per-network, per-variant models using subclass algorithm
        for network in NetworkDict().keys():
            self.models['connectome'][network] = self.fit_model(connectome[network], y)
            self.models['covariates'][network] = LinearRegression().fit(covariates, y)
            self.models['residuals'][network] = LinearRegression().fit(residuals[network], y)
            self.models['full'][network] = self.fit_model(
                np.concatenate([connectome[network], covariates], axis=1), y
            )

        return self

    def predict(self, X: np.ndarray, covariates: np.ndarray) -> ModelDict:
        """
        Predict y for connectome, covariates, residuals, and full models.
        """
        connectome = {}
        residuals = {}
        network_strengths = {}

        for network in ['positive', 'negative']:
            network_strengths[network] = np.sum(X[:, self.edges[network]], axis=1).reshape(-1, 1)
            residuals[network] = network_strengths[network] - self.models_residuals[network].predict(covariates)
            if self.edges[network].shape[0] == 0:
                connectome[network] = np.zeros((X.shape[0], 1))
            else:
                connectome[network] = X[:, self.edges[network]]

        residuals['both'] = np.hstack((residuals['positive'], residuals['negative']))
        network_strengths['both'] = np.hstack((network_strengths['positive'], network_strengths['negative']))
        if np.concatenate([X[:, self.edges['positive']], X[:, self.edges['negative']]], axis=1).shape[1] == 0:
            connectome['both'] = np.zeros((X.shape[0], 1))
        else:
            connectome['both'] = np.concatenate([X[:, self.edges['positive']], X[:, self.edges['negative']]], axis=1)

        predictions = ModelDict()
        for network in ['positive', 'negative', 'both']:
            predictions['connectome'][network] = self.predict_model(self.models['connectome'][network],
                                                                   connectome[network])
            predictions['covariates'][network]  = self.models['covariates'][network].predict(covariates)
            predictions['residuals'][network]   = self.models['residuals'][network].predict(residuals[network])
            predictions['full'][network]        = self.predict_model(self.models['full'][network],
                                                                     np.concatenate([connectome[network], covariates], axis=1))
        return predictions

    def predict_model(self, model, X: np.ndarray) -> np.ndarray:
        return model.predict(X)

    def get_network_strengths(self, X: np.ndarray, covariates: np.ndarray):
        connectome = {}
        residuals = {}
        for network in ['positive', 'negative']:
            connectome[network] = np.sum(X[:, self.edges[network]], axis=1).reshape(-1, 1)
            residuals[network] = connectome[network] - self.models_residuals[network].predict(covariates)
        return {"connectome": connectome, "residuals": residuals}


# --- Concrete CPMs that only override the modeling bits ---

class LinearCPMModel(BaseCPMModel):
    """
    CPM using ordinary least squares for the predictive pieces.
    """
    def __init__(self, edges: Dict[str, np.ndarray], **linreg_kwargs):
        super().__init__(edges)
        self.linreg_kwargs = linreg_kwargs
        self.name = 'LinearCPMModel'

    def fit_model(self, X: np.ndarray, y: np.ndarray):
        return LinearRegression(**self.linreg_kwargs).fit(X, y)


class DecisionTreeCPMModel(BaseCPMModel):
    """
    CPM using a DecisionTreeRegressor for the predictive pieces.
    """
    name = 'DecisionTreeCPMModel'
    def __init__(self, edges: Dict[str, np.ndarray], **tree_kwargs):
        super().__init__(edges)
        # Sensible defaults; can be overridden via **tree_kwargs
        defaults = dict(random_state=0)
        defaults.update(tree_kwargs)
        self.tree_kwargs = defaults

    def fit_model(self, X: np.ndarray, y: np.ndarray):
        return DecisionTreeRegressor(**self.tree_kwargs).fit(X, y)


class RandomForestCPMModel(BaseCPMModel):
    """
    CPM using a RandomForestRegressor for the predictive pieces.
    """
    name = 'RandomForestCPMModel'
    def __init__(self, edges, **rf_kwargs):
        super().__init__(edges)
        # Sensible defaults; you can override via **rf_kwargs
        defaults = dict(
            n_estimators=50,
            random_state=0,
            n_jobs=-1,
        )
        defaults.update(rf_kwargs)
        self.rf_kwargs = defaults

    def fit_model(self, X, y):
        return RandomForestRegressor(**self.rf_kwargs).fit(X, y)


class GAMCPMModel(BaseCPMModel):
    name = 'GAMCPMModel'
    def __init__(self, edges, **gam_kwargs):
        super().__init__(edges)
        # Sensible defaults; you can override via **rf_kwargs
        defaults = dict(

        )
        defaults.update(gam_kwargs)
        self.gam_kwargs = defaults

    def fit_model(self, X, y):
        return LinearGAM(**self.gam_kwargs).fit(X, y)
