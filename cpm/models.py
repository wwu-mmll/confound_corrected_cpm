import numpy as np
from sklearn.linear_model import LinearRegression


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


class LinearCPMModel:
    def __init__(self, edges):
        self.models = ModelDict()
        self.models_residuals = {}
        self.edges = edges

    def fit(self, X, y, covariates):
        connectome = {}
        residuals = {}
        for network in ['positive', 'negative']:
            # Compute sum_positive and sum_negative
            connectome[network] = np.sum(X[:, self.edges[network]], axis=1).reshape(-1, 1)
            self.models_residuals[network] = LinearRegression().fit(covariates, connectome[network])
            residuals[network] = connectome[network] - self.models_residuals[network].predict(covariates)

        residuals['both'] = np.hstack((residuals['positive'], residuals['negative']))
        connectome['both'] = np.hstack((connectome['positive'], connectome['negative']))

        for network in NetworkDict().keys():
            self.models['connectome'][network] = LinearRegression().fit(connectome[network], y)
            self.models['covariates'][network] = LinearRegression().fit(covariates, y)
            self.models['residuals'][network] = LinearRegression().fit(residuals[network], y)
            self.models['full'][network] = LinearRegression().fit(np.hstack((connectome[network], covariates)), y)

        return self

    def predict(self, X, covariates):
        connectome = {}
        residuals = {}
        for network in ['positive', 'negative']:
            # Compute sum_positive and sum_negative
            connectome[network] = np.sum(X[:, self.edges[network]], axis=1).reshape(-1, 1)
            residuals[network] = connectome[network] - self.models_residuals[network].predict(covariates)

        residuals['both'] = np.hstack((residuals['positive'], residuals['negative']))
        connectome['both'] = np.hstack((connectome['positive'], connectome['negative']))

        predictions = ModelDict()
        for network in ['positive', 'negative', 'both']:
            predictions['connectome'][network] = self.models['connectome'][network].predict(connectome[network])
            predictions['covariates'][network] = self.models['covariates'][network].predict(covariates)
            predictions['residuals'][network] = self.models['residuals'][network].predict(residuals[network])
            predictions['full'][network] = self.models['full'][network].predict(np.hstack((connectome[network], covariates)))

        return predictions
