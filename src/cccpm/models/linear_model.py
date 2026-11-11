"""PyTorch linear CPM backend (OLS / logistic regression)."""
import numpy as np
import torch

from cccpm.constants import Networks, Models, TaskType


class LinearCPM:
    """PyTorch CPM model: OLS for regression, logistic regression (IRLS) for binary classification.

    Every model variant, hyperparameter configuration, and permutation run is
    fit in one batched linear-algebra pass rather than a Python loop.
    Whether the covariate variance has been taken out of ``X`` is decided by
    the caller (``CPMAnalysis(model_input=...)``), not here -- this class
    fits whatever connectivity it is given, which is what keeps deconfounding
    a property of the run rather than an extra model. Without covariates only
    ``connectome`` is defined; ``covariates``/``full``/``increment`` are
    filled with NaN rather than a number that looks meaningful (see
    ``available_models``).
    """
    name = "LinearCPM"

    def __init__(self, edges, device='cpu', task_type=TaskType.regression):
        self.device = torch.device(device)
        self.task_type = task_type

        edges = torch.as_tensor(edges, device=self.device)
        if edges.dim() != 3:
            raise ValueError(
                f"edges must be [N_features, 2, N_runs], got shape {tuple(edges.shape)}")
        self.edges = edges

        self.coefs = {}
        self.has_covariates = None

    @staticmethod
    def _covariates_present(covariates):
        """An absent covariate design is either None or zero-width."""
        if covariates is None:
            return False
        return int(np.shape(covariates)[1]) > 0

    @property
    def available_models(self):
        """Model variants this fit actually defines, by name."""
        if self.has_covariates is False:
            return [Models.connectome.name]
        return [m.name for m in Models]

    def _network_strengths(self, X):
        """X [N, F] + edges [F, 2, R] -> (pos, neg), each [R, N]."""
        pos = torch.einsum('nf,fr->rn', X, self.edges[:, Networks.positive].float())
        neg = torch.einsum('nf,fr->rn', X, self.edges[:, Networks.negative].float())
        return pos, neg

    def _expand_covariates(self, cov, n_runs):
        """cov [N, C] -> [R, N, C], broadcast across runs."""
        n_samples, n_cov = cov.shape
        return cov.view(1, n_samples, n_cov).expand(n_runs, n_samples, n_cov)

    def fit(self, X, y, covariates=None):
        """Fit all CPM variants for every hyperparameter configuration and run."""
        X = torch.as_tensor(X, device=self.device, dtype=torch.float32)
        y = torch.as_tensor(y, device=self.device, dtype=torch.float32)
        self.has_covariates = self._covariates_present(covariates)

        n_runs = self.edges.shape[2]
        if y.shape[1] != n_runs:
            raise ValueError(f"y has {y.shape[1]} runs but edges has {n_runs}")
        R = n_runs

        pos_str, neg_str = self._network_strengths(X)

        solve = (self._solve_logistic if self.task_type == TaskType.classification
                 else self._solve_ols)

        y_runs = y.t()
        y_batch = y_runs.unsqueeze(-1)

        conn = {
            'positive': pos_str.unsqueeze(-1),
            'negative': neg_str.unsqueeze(-1),
            'both': torch.stack([pos_str, neg_str], dim=-1),
        }

        if not self.has_covariates:
            for net, X_conn in conn.items():
                self.coefs[f'connectome_{net}'] = solve(X_conn, y_batch)
            return self

        cov = torch.as_tensor(covariates, device=self.device, dtype=torch.float32)
        cov_b = self._expand_covariates(cov, R)

        self.coefs['covariates'] = solve(cov_b, y_runs.unsqueeze(-1))

        for net, X_conn in conn.items():
            X_full = torch.cat([X_conn, cov_b], dim=-1)
            self.coefs[f'connectome_{net}'] = solve(X_conn, y_batch)
            self.coefs[f'full_{net}'] = solve(X_full, y_batch)

        return self

    def predict(self, X, covariates=None, return_proba=False):
        """Predicts with every fitted variant -> [N_samples, N_models, N_networks, N_runs]."""
        X = torch.as_tensor(X, device=self.device, dtype=torch.float32)
        R, N = self.edges.shape[2], X.shape[0]

        pos_str, neg_str = self._network_strengths(X)

        predictions = torch.zeros(N, len(Models), len(Networks), R,
                                  device=self.device, dtype=torch.float32)

        conn = [
            (Networks.positive, pos_str.unsqueeze(-1)),
            (Networks.negative, neg_str.unsqueeze(-1)),
            (Networks.both, torch.stack([pos_str, neg_str], dim=-1)),
        ]

        if not self.has_covariates:
            # Undefined variants get NaN, not a number that would read as a real model score.
            for model in (Models.covariates, Models.full, Models.increment):
                predictions[:, model] = float('nan')
            for net_idx, X_conn in conn:
                predictions[:, Models.connectome, net_idx] = self._predict_linear(
                    X_conn, self.coefs[f'connectome_{net_idx.name}']).squeeze(-1).t()
        else:
            cov = torch.as_tensor(covariates, device=self.device, dtype=torch.float32)
            cov_b = self._expand_covariates(cov, R)

            cov_pred = self._predict_linear(cov_b, self.coefs['covariates'])
            cov_pred = cov_pred.squeeze(-1).t()
            predictions[:, Models.covariates] = cov_pred.unsqueeze(1).expand(N, len(Networks), R)

            for net_idx, X_conn in conn:
                X_full = torch.cat([X_conn, cov_b], dim=-1)
                name = net_idx.name
                predictions[:, Models.connectome, net_idx] = self._predict_linear(
                    X_conn, self.coefs[f'connectome_{name}']).squeeze(-1).t()
                predictions[:, Models.full, net_idx] = self._predict_linear(
                    X_full, self.coefs[f'full_{name}']).squeeze(-1).t()

        if self.task_type == TaskType.classification:
            predictions = torch.sigmoid(predictions)
            if not return_proba:
                # `nan > 0.5` is False, which would read as a confident class 0; keep NaN NaN.
                labels = (predictions > 0.5).float()
                predictions = torch.where(torch.isnan(predictions),
                                          predictions, labels)

        return predictions

    def predict_proba(self, X, covariates=None):
        """Probabilities for classification; identical to predict() for regression."""
        return self.predict(X, covariates, return_proba=True)

    def predict_class(self, X, covariates=None):
        """Class labels for classification; identical to predict() for regression."""
        return self.predict(X, covariates, return_proba=False)

    @staticmethod
    def _add_intercept(X):
        ones = torch.ones(*X.shape[:-1], 1, device=X.device, dtype=X.dtype)
        return torch.cat([ones, X], dim=-1)

    def _solve_ols(self, X, y):
        """OLS via the normal equations with a small ridge for conditioning."""
        X_design = self._add_intercept(X)
        XtX = torch.matmul(X_design.transpose(-1, -2), X_design)
        XtX.diagonal(dim1=-2, dim2=-1).add_(1e-8)
        Xty = torch.matmul(X_design.transpose(-1, -2), y)
        return torch.linalg.solve(XtX, Xty)

    def _solve_logistic(self, X, y, max_iter=25, tol=1e-6):
        """Logistic regression via IRLS, initialised from the OLS solution."""
        X_design = self._add_intercept(X)
        beta = self._solve_ols(X, y)

        for _ in range(max_iter):
            logits = torch.matmul(X_design, beta)
            p = torch.sigmoid(logits)
            W = (p * (1 - p)).clamp(min=1e-8)
            z = logits + (y - p) / W

            sqrt_W = torch.sqrt(W)
            X_w = X_design * sqrt_W
            z_w = z * sqrt_W

            XtX = torch.matmul(X_w.transpose(-1, -2), X_w)
            XtX.diagonal(dim1=-2, dim2=-1).add_(1e-8)
            Xtz = torch.matmul(X_w.transpose(-1, -2), z_w)

            beta_new = torch.linalg.solve(XtX, Xtz)
            converged = torch.max(torch.abs(beta_new - beta)) < tol
            beta = beta_new
            if converged:
                break

        return beta

    def _predict_linear(self, X, beta):
        """X [..., N, F], beta [..., F+1, 1] -> [..., N, 1]."""
        return torch.matmul(self._add_intercept(X), beta)

    def get_network_strengths(self, X: np.ndarray, covariates: np.ndarray = None):
        """Per-subject positive/negative network strengths of X, for the HTML report."""
        X = torch.as_tensor(X, device=self.device, dtype=torch.float32)

        pos_str, neg_str = self._network_strengths(X)
        pos_str, neg_str = pos_str.t(), neg_str.t()

        return {
            "connectome": {
                Networks.positive.name: pos_str,
                Networks.negative.name: neg_str,
            },
        }
