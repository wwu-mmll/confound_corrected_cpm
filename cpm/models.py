import numpy as np
from sklearn.linear_model import LinearRegression


class LinearCPMModel:
    def __init__(self, significant_edges, use_covariates=True):
        self.lm = None
        self.use_covariates = use_covariates
        self.significant_edges = significant_edges

    def fit(self, X, y, covariates):
        self.lm = LinearRegression().fit(np.sum(X[:, self.significant_edges], axis=1).reshape(-1, 1), y)
        return self

    def predict(self, X, covariates):
        return self.lm.predict(np.sum(X[:, self.significant_edges], axis=1).reshape(-1, 1))


class LinearCPMModelv2:
    def __init__(self, positive_edges, negative_edges, calculate_increments=True):
        self.full_lm = dict()
        self.connectome_lm = dict()
        self.covariates_lm = dict()
        self.increments = calculate_increments
        self.positive_edges = positive_edges
        self.negative_edges = negative_edges

    def fit(self, X, y, covariates):
        sum_positive = np.sum(X[:, self.positive_edges], axis=1).reshape(-1, 1)
        sum_negative = np.sum(X[:, self.negative_edges], axis=1).reshape(-1, 1)
        self.connectome_lm['positive'] = LinearRegression().fit(sum_positive, y)
        self.connectome_lm['negative'] = LinearRegression().fit(sum_negative, y)
        self.connectome_lm['both'] = LinearRegression().fit(sum_positive - sum_negative, y)

        self.covariates_lm['positive'] = LinearRegression().fit(covariates, y)
        self.covariates_lm['negative'] = LinearRegression().fit(covariates, y)
        self.covariates_lm['both'] = LinearRegression().fit(covariates, y)

        self.full_lm['positive'] = LinearRegression().fit(np.concatenate([sum_positive, covariates], axis=1), y)
        self.full_lm['negative'] = LinearRegression().fit(np.concatenate([sum_negative, covariates], axis=1), y)
        self.full_lm['both'] = LinearRegression().fit(np.concatenate([sum_positive - sum_negative, covariates], axis=1), y)

        return self

    def predict(self, X, covariates):
        sum_positive = np.sum(X[:, self.positive_edges], axis=1).reshape(-1, 1)
        sum_negative = np.sum(X[:, self.negative_edges], axis=1).reshape(-1, 1)

        preds_positive_connectome = self.connectome_lm['positive'].predict(sum_positive)
        preds_negative_connectome = self.connectome_lm['negative'].predict(sum_negative)
        preds_both_connectome = self.connectome_lm['both'].predict(sum_positive)

        preds_positive_cov = self.covariates_lm['positive'].predict(covariates)
        preds_negative_cov = self.covariates_lm['negative'].predict(covariates)
        preds_both_cov = self.covariates_lm['both'].predict(covariates)

        preds_positive_full = self.full_lm['positive'].predict(np.concatenate([sum_positive, covariates], axis=1))
        preds_negative_full = self.full_lm['negative'].predict(np.concatenate([sum_negative, covariates], axis=1))
        preds_both_full = self.full_lm['both'].predict(np.concatenate([sum_positive - sum_negative, covariates], axis=1))

        return {'connectome': {'positive': preds_positive_connectome, 'negative': preds_negative_connectome, 'both': preds_both_connectome},
                'covariates': {'positive': preds_positive_cov, 'negative': preds_negative_cov,
                               'both': preds_both_cov},
                'full': {'positive': preds_positive_full, 'negative': preds_negative_full, 'both': preds_both_full}}
