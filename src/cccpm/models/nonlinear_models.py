from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from cccpm.constants import Networks, Models, TaskType


class BaseCPM(ABC):
    """
    Base class for non-linear CPM models.

    Matches the tensor-based pipeline interface of LinearCPM but delegates
    the connectome and full model fitting to subclass-defined estimators
    (sklearn, pygam, etc.).  Covariates and residuals models always use
    ordinary least-squares (LinearRegression).

    Constructor
    -----------
    edges : tensor [N_features, 2, N_runs]
    device : str ('cpu' or 'cuda') — used only for output tensors
    task_type : TaskType — stored but not currently used (subclasses are
                regression-only; classification uses LinearCPM's IRLS)
    """

    name = "BaseCPM"

    def __init__(self, edges, device='cpu', task_type=TaskType.regression):
        self.device = torch.device(device)
        self.task_type = task_type
        self.edges = torch.as_tensor(edges, device=self.device)
        self._fitted = []
        self._resid_models = []

    # ------------------------------------------------------------------
    # Abstract / overridable
    # ------------------------------------------------------------------

    @abstractmethod
    def fit_model(self, X, y):
        """Return a fitted estimator for (X, y).  *X* and *y* are numpy."""

    def predict_model(self, estimator, X):
        """Predict with a fitted estimator.  Override for non-sklearn APIs."""
        return estimator.predict(X)

    # ------------------------------------------------------------------
    # Pipeline interface
    # ------------------------------------------------------------------

    def fit(self, X, y, covariates):
        """
        Fit all CPM model variants for every permutation run.

        Parameters
        ----------
        X : array-like [N_samples, N_features]
        y : array-like [N_samples, N_runs]
        covariates : array-like [N_samples, N_cov]
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        covariates = np.asarray(covariates, dtype=np.float32)

        edges_np = self.edges.cpu().numpy()
        n_samples, n_runs = y.shape

        self._fitted = []
        self._resid_models = []

        for run in range(n_runs):
            run_models = {}
            run_resid = {}

            pos_mask = edges_np[:, Networks.positive, run].astype(bool)
            neg_mask = edges_np[:, Networks.negative, run].astype(bool)

            pos_str = X[:, pos_mask].sum(axis=1, keepdims=True) if pos_mask.any() else np.zeros((n_samples, 1), dtype=np.float32)
            neg_str = X[:, neg_mask].sum(axis=1, keepdims=True) if neg_mask.any() else np.zeros((n_samples, 1), dtype=np.float32)

            run_resid['pos'] = LinearRegression().fit(covariates, pos_str)
            run_resid['neg'] = LinearRegression().fit(covariates, neg_str)

            pos_resid = pos_str - run_resid['pos'].predict(covariates)
            neg_resid = neg_str - run_resid['neg'].predict(covariates)

            y_run = y[:, run]

            feats = {
                'positive': {'conn': pos_str, 'resid': pos_resid},
                'negative': {'conn': neg_str, 'resid': neg_resid},
                'both': {
                    'conn': np.hstack([pos_str, neg_str]),
                    'resid': np.hstack([pos_resid, neg_resid]),
                },
            }

            # Covariates model — same for all networks, fit once
            run_models['covariates'] = LinearRegression().fit(covariates, y_run)

            for net in ['positive', 'negative', 'both']:
                run_models[f'connectome_{net}'] = self.fit_model(feats[net]['conn'], y_run)
                run_models[f'residuals_{net}'] = LinearRegression().fit(feats[net]['resid'], y_run)
                X_full = np.hstack([feats[net]['conn'], covariates])
                run_models[f'full_{net}'] = self.fit_model(X_full, y_run)

            self._fitted.append(run_models)
            self._resid_models.append(run_resid)

        return self

    def predict(self, X, covariates, return_proba=False):
        """
        Predict for all runs.

        Returns
        -------
        torch.Tensor [N_samples, N_models, N_networks, N_runs]
        """
        X = np.asarray(X, dtype=np.float32)
        covariates = np.asarray(covariates, dtype=np.float32)

        edges_np = self.edges.cpu().numpy()
        n_samples = X.shape[0]
        n_runs = self.edges.size(2)

        predictions = np.zeros(
            (n_samples, len(Models), len(Networks), n_runs), dtype=np.float32
        )

        for run in range(n_runs):
            pos_mask = edges_np[:, Networks.positive, run].astype(bool)
            neg_mask = edges_np[:, Networks.negative, run].astype(bool)

            pos_str = X[:, pos_mask].sum(axis=1, keepdims=True) if pos_mask.any() else np.zeros((n_samples, 1), dtype=np.float32)
            neg_str = X[:, neg_mask].sum(axis=1, keepdims=True) if neg_mask.any() else np.zeros((n_samples, 1), dtype=np.float32)

            pos_resid = pos_str - self._resid_models[run]['pos'].predict(covariates)
            neg_resid = neg_str - self._resid_models[run]['neg'].predict(covariates)

            feats = {
                'positive': {'conn': pos_str, 'resid': pos_resid},
                'negative': {'conn': neg_str, 'resid': neg_resid},
                'both': {
                    'conn': np.hstack([pos_str, neg_str]),
                    'resid': np.hstack([pos_resid, neg_resid]),
                },
            }

            nets = [
                ('positive', Networks.positive),
                ('negative', Networks.negative),
                ('both', Networks.both),
            ]

            for net_name, net_idx in nets:
                predictions[:, Models.covariates, net_idx, run] = (
                    self.predict_model(self._fitted[run]['covariates'], covariates).ravel()
                )
                predictions[:, Models.connectome, net_idx, run] = (
                    self.predict_model(self._fitted[run][f'connectome_{net_name}'], feats[net_name]['conn']).ravel()
                )
                predictions[:, Models.residuals, net_idx, run] = (
                    self.predict_model(self._fitted[run][f'residuals_{net_name}'], feats[net_name]['resid']).ravel()
                )
                X_full = np.hstack([feats[net_name]['conn'], covariates])
                predictions[:, Models.full, net_idx, run] = (
                    self.predict_model(self._fitted[run][f'full_{net_name}'], X_full).ravel()
                )

        return torch.from_numpy(predictions).to(self.device)

    def get_network_strengths(self, X, covariates):
        """
        Calculate network strengths for all runs.

        Returns
        -------
        dict with "connectome" and "residuals" sub-dicts, each mapping
        "positive"/"negative" to tensors [N_samples, N_runs].
        """
        X = np.asarray(X, dtype=np.float32)
        covariates = np.asarray(covariates, dtype=np.float32)

        edges_np = self.edges.cpu().numpy()
        n_samples = X.shape[0]
        n_runs = self.edges.size(2)

        pos_strengths = np.zeros((n_samples, n_runs), dtype=np.float32)
        neg_strengths = np.zeros((n_samples, n_runs), dtype=np.float32)
        pos_residuals = np.zeros((n_samples, n_runs), dtype=np.float32)
        neg_residuals = np.zeros((n_samples, n_runs), dtype=np.float32)

        for run in range(n_runs):
            pos_mask = edges_np[:, Networks.positive, run].astype(bool)
            neg_mask = edges_np[:, Networks.negative, run].astype(bool)

            pos_str = X[:, pos_mask].sum(axis=1) if pos_mask.any() else np.zeros(n_samples, dtype=np.float32)
            neg_str = X[:, neg_mask].sum(axis=1) if neg_mask.any() else np.zeros(n_samples, dtype=np.float32)

            pos_strengths[:, run] = pos_str
            neg_strengths[:, run] = neg_str
            pos_residuals[:, run] = pos_str - self._resid_models[run]['pos'].predict(covariates).ravel()
            neg_residuals[:, run] = neg_str - self._resid_models[run]['neg'].predict(covariates).ravel()

        return {
            "connectome": {
                Networks.positive.name: torch.from_numpy(pos_strengths).to(self.device),
                Networks.negative.name: torch.from_numpy(neg_strengths).to(self.device),
            },
            "residuals": {
                Networks.positive.name: torch.from_numpy(pos_residuals).to(self.device),
                Networks.negative.name: torch.from_numpy(neg_residuals).to(self.device),
            },
        }


class DecisionTreeCPM(BaseCPM):
    name = "DecisionTreeCPM"

    def __init__(self, edges, device='cpu', task_type=TaskType.regression, **tree_kwargs):
        super().__init__(edges, device, task_type)
        defaults = dict(random_state=0)
        defaults.update(tree_kwargs)
        self.tree_kwargs = defaults

    def fit_model(self, X, y):
        return DecisionTreeRegressor(**self.tree_kwargs).fit(X, y)


class RandomForestCPM(BaseCPM):
    name = "RandomForestCPM"

    def __init__(self, edges, device='cpu', task_type=TaskType.regression, **rf_kwargs):
        super().__init__(edges, device, task_type)
        defaults = dict(n_estimators=50, random_state=0, n_jobs=-1)
        defaults.update(rf_kwargs)
        self.rf_kwargs = defaults

    def fit_model(self, X, y):
        return RandomForestRegressor(**self.rf_kwargs).fit(X, y)


class GAMCPM(BaseCPM):
    name = "GAMCPM"

    def __init__(self, edges, device='cpu', task_type=TaskType.regression, **gam_kwargs):
        super().__init__(edges, device, task_type)
        self.gam_kwargs = gam_kwargs

    def fit_model(self, X, y):
        from pygam import LinearGAM
        return LinearGAM(**self.gam_kwargs).fit(X, y)

    def predict_model(self, estimator, X):
        return estimator.predict(X)
