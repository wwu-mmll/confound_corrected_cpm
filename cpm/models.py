import numpy as np
from sklearn.linear_model import LinearRegression


class LinearCPMModel:
    def __init__(self, significant_edges):
        self.lm = None
        self.significant_edges = significant_edges

    def fit(self, X, y, covariates):
        self.lm = LinearRegression().fit(np.sum(X[:, self.significant_edges], axis=1).reshape(-1, 1), y)
        return self

    def predict(self, X, covariates):
        return self.lm.predict(np.sum(X[:, self.significant_edges], axis=1).reshape(-1, 1))
