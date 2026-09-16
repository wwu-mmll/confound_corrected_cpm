import numpy as np
import torch

from cccpm.constants import Networks, Models, TaskType


class LinearCPM:
    """
    PyTorch CPM model: OLS for regression, logistic regression (IRLS) for
    binary classification.

    Every model variant, hyperparameter configuration and permutation is fitted
    in one batched linear-algebra pass rather than a Python loop.

    Shapes
    ------
    X          : [N_samples, N_features]
    y          : [N_samples, N_runs]          (one column per permutation)
    covariates : [N_samples, N_cov]
    edges      : [N_features, 2, N_runs]      dim 1 = [positive, negative]

    Predictions come back as [N_samples, N_models, N_networks, N_runs].
    """
    name = "LinearCPM"

    def __init__(self, edges, device='cpu', task_type=TaskType.regression):
        """
        Args:
            edges: [N_features, 2, N_runs] boolean mask tensor.
            device: 'cpu' or 'cuda'
            task_type: TaskType.regression or TaskType.classification
        """
        self.device = torch.device(device)
        self.task_type = task_type

        edges = torch.as_tensor(edges, device=self.device)
        if edges.dim() != 3:
            raise ValueError(
                f"edges must be [N_features, 2, N_runs], got shape {tuple(edges.shape)}")
        self.edges = edges

        self.coefs = {}         # fitted coefficients per model variant
        self.resid_models = {}  # covariate models used to residualise strengths

    # --- shape helpers ----------------------------------------------------

    def _network_strengths(self, X):
        """X [N, F] + edges [F, 2, R] -> (pos, neg), each [R, N]."""
        pos = torch.einsum('nf,fr->rn', X, self.edges[:, Networks.positive].float())
        neg = torch.einsum('nf,fr->rn', X, self.edges[:, Networks.negative].float())
        return pos, neg

    def _expand_covariates(self, cov, n_runs):
        """cov [N, C] -> [R, N, C], broadcast across runs."""
        n_samples, n_cov = cov.shape
        return cov.view(1, n_samples, n_cov).expand(n_runs, n_samples, n_cov)

    # --- fit / predict ----------------------------------------------------

    def fit(self, X, y, covariates):
        """Fit all CPM variants for every hyperparameter configuration and run."""
        X = torch.as_tensor(X, device=self.device, dtype=torch.float32)
        y = torch.as_tensor(y, device=self.device, dtype=torch.float32)
        cov = torch.as_tensor(covariates, device=self.device, dtype=torch.float32)

        n_samples, n_cov = cov.shape
        n_runs = self.edges.shape[2]
        if y.shape[1] != n_runs:
            raise ValueError(f"y has {y.shape[1]} runs but edges has {n_runs}")
        R, N = n_runs, n_samples

        pos_str, neg_str = self._network_strengths(X)          # [R, N]
        cov_b = self._expand_covariates(cov, R)                # [R, N, C]

        solve = (self._solve_logistic if self.task_type == TaskType.classification
                 else self._solve_ols)

        # Residualisers: network strength ~ covariates. Always OLS -- these
        # remove covariate variance from a continuous strength, independent of
        # whether the outcome is continuous or binary.
        self.resid_models['pos'] = self._solve_ols(cov_b, pos_str.unsqueeze(-1))
        self.resid_models['neg'] = self._solve_ols(cov_b, neg_str.unsqueeze(-1))
        pos_resid = pos_str - self._predict_linear(cov_b, self.resid_models['pos']).squeeze(-1)
        neg_resid = neg_str - self._predict_linear(cov_b, self.resid_models['neg']).squeeze(-1)

        feats = {
            'positive': (pos_str.unsqueeze(-1), pos_resid.unsqueeze(-1)),
            'negative': (neg_str.unsqueeze(-1), neg_resid.unsqueeze(-1)),
            'both': (torch.stack([pos_str, neg_str], dim=-1),
                     torch.stack([pos_resid, neg_resid], dim=-1)),
        }

        y_runs = y.t()                                         # [R, N]
        self.coefs['covariates'] = solve(cov_b, y_runs.unsqueeze(-1))  # [R, C+1, 1]

        y_batch = y_runs.unsqueeze(-1)                         # [R, N, 1]

        for net, (X_conn, X_resid) in feats.items():
            X_full = torch.cat([X_conn, cov_b], dim=-1)
            self.coefs[f'connectome_{net}'] = solve(X_conn, y_batch)
            self.coefs[f'residuals_{net}'] = solve(X_resid, y_batch)
            self.coefs[f'full_{net}'] = solve(X_full, y_batch)

        return self

    def predict(self, X, covariates, return_proba=False):
        """
        Predict with every fitted variant.

        Returns [N_samples, N_models, N_networks, N_params, N_runs]. For
        classification, probabilities when ``return_proba`` else 0/1 labels.
        """
        X = torch.as_tensor(X, device=self.device, dtype=torch.float32)
        cov = torch.as_tensor(covariates, device=self.device, dtype=torch.float32)

        n_samples, n_cov = cov.shape
        R, N = self.edges.shape[2], n_samples

        pos_str, neg_str = self._network_strengths(X)
        cov_b = self._expand_covariates(cov, R)

        pos_resid = pos_str - self._predict_linear(cov_b, self.resid_models['pos']).squeeze(-1)
        neg_resid = neg_str - self._predict_linear(cov_b, self.resid_models['neg']).squeeze(-1)

        predictions = torch.zeros(N, len(Models), len(Networks), R,
                                  device=self.device, dtype=torch.float32)

        # Covariates model: same for every network.
        cov_pred = self._predict_linear(cov_b, self.coefs['covariates'])   # [R, N, 1]
        cov_pred = cov_pred.squeeze(-1).t()                                # [N, R]
        predictions[:, Models.covariates] = cov_pred.unsqueeze(1).expand(N, len(Networks), R)

        networks = [
            (Networks.positive, pos_str.unsqueeze(-1), pos_resid.unsqueeze(-1)),
            (Networks.negative, neg_str.unsqueeze(-1), neg_resid.unsqueeze(-1)),
            (Networks.both, torch.stack([pos_str, neg_str], dim=-1),
             torch.stack([pos_resid, neg_resid], dim=-1)),
        ]
        for net_idx, X_conn, X_resid in networks:
            X_full = torch.cat([X_conn, cov_b], dim=-1)
            name = net_idx.name
            # each _predict_linear -> [R, N, 1]; reorder to [N, R]
            predictions[:, Models.connectome, net_idx] = self._predict_linear(
                X_conn, self.coefs[f'connectome_{name}']).squeeze(-1).t()
            predictions[:, Models.residuals, net_idx] = self._predict_linear(
                X_resid, self.coefs[f'residuals_{name}']).squeeze(-1).t()
            predictions[:, Models.full, net_idx] = self._predict_linear(
                X_full, self.coefs[f'full_{name}']).squeeze(-1).t()

        if self.task_type == TaskType.classification:
            predictions = torch.sigmoid(predictions)
            if not return_proba:
                predictions = (predictions > 0.5).float()

        return predictions

    def predict_proba(self, X, covariates):
        """Probabilities for classification; identical to predict() for regression."""
        return self.predict(X, covariates, return_proba=True)

    def predict_class(self, X, covariates):
        """Class labels for classification; identical to predict() for regression."""
        return self.predict(X, covariates, return_proba=False)

    # --- solvers ----------------------------------------------------------
    #
    # All operate on [..., N, F] designs with arbitrary leading batch axes, and
    # prepend an intercept column. Returned coefficients are [..., F+1, 1].

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

    # --- reporting --------------------------------------------------------

    def get_network_strengths(self, X: np.ndarray, covariates: np.ndarray):
        """
        Per-subject positive/negative network strengths, raw and residualised,
        for the HTML report.

        Returns a dict of [N_samples, N_runs] tensors.
        """
        X = torch.as_tensor(X, device=self.device, dtype=torch.float32)
        cov = torch.as_tensor(covariates, device=self.device, dtype=torch.float32)

        R = self.edges.shape[2]

        pos_str, neg_str = self._network_strengths(X)        # [R, N]
        pos_str, neg_str = pos_str.t(), neg_str.t()          # [N, R]

        cov_runs = self._expand_covariates(cov, R)
        pred_pos = self._predict_linear(cov_runs, self.resid_models['pos']).squeeze(-1).t()
        pred_neg = self._predict_linear(cov_runs, self.resid_models['neg']).squeeze(-1).t()

        return {
            "connectome": {
                Networks.positive.name: pos_str,
                Networks.negative.name: neg_str,
            },
            "residuals": {
                Networks.positive.name: pos_str - pred_pos,
                Networks.negative.name: neg_str - pred_neg,
            },
        }
