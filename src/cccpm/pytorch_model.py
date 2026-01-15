import numpy as np
import torch

from cccpm.constants import Networks, Models


class LinearCPMModel:
    """
    A PyTorch implementation of CPM optimized for speed.

    Optimizations:
    1. Vectorized Over Permutations: Fits all N_perms models in parallel.
    2. Fast Cholesky Solver: Uses Normal Equations (XtX^-1 Xty) instead of SVD/QR.
    3. Shared Covariate Logic: Handles fixed covariates efficiently.

    Input Shapes:
      - X: [N_samples, N_features]
      - y: [N_samples, N_perms]
    """
    name = "LinearCPMTorch"

    def __init__(self, edges, device='cuda'):
        """
        Args:
            edges: Boolean masks tensor with shape [N_features, 2, N_runs].
                   Dimension 1 corresponds to [Positive, Negative].
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device)

        self.edges = torch.as_tensor(edges, device=self.device)

        self.coefs = {}  # Stores coefficients (Beta) for prediction
        self.resid_models = {}  # Stores models used to residualize strengths

    def fit(self, X, y, covariates):
        """
        Fits all CPM variations (Connectome, Covariates, Residuals, Full) for all permutations.
        Args:
            X: [N_samples, N_features]
            y: [N_samples, N_perms]
            covariates: [N_samples, N_cov]
        """
        # 1. Setup & Device Transfer
        X = torch.as_tensor(X, device=self.device, dtype=torch.float32)
        y = torch.as_tensor(y, device=self.device, dtype=torch.float32)
        cov = torch.as_tensor(covariates, device=self.device, dtype=torch.float32)

        n_samples, n_perms = y.shape

        # 2. Calculate Network Strengths (Matrix Multiplication)
        # X: [N_samples, N_features] @ edges: [N_features, N_runs] -> [N_samples, N_runs]
        pos_str = torch.matmul(X, self.edges[:, Networks.positive, :].float())
        neg_str = torch.matmul(X, self.edges[:, Networks.negative, :].float())

        # 3. Fit Residualizers (Strength ~ Covariates)
        # Remove covariate variance from strengths.
        # This uses Shared Solver (X is fixed).
        self.resid_models['pos'] = self._solve_shared(cov, pos_str)
        self.resid_models['neg'] = self._solve_shared(cov, neg_str)

        pos_resid = pos_str - self._pred_shared(cov, self.resid_models['pos'])
        neg_resid = neg_str - self._pred_shared(cov, self.resid_models['neg'])

        # 4. Prepare Feature Sets
        # 'conn' = raw strength, 'resid' = residualized strength
        feats = {
            'positive': {'conn': pos_str, 'resid': pos_resid},
            'negative': {'conn': neg_str, 'resid': neg_resid},
            'both': {'conn': torch.stack([pos_str, neg_str], dim=2),  # [N, P, 2]
                     'resid': torch.stack([pos_resid, neg_resid], dim=2)}
        }

        # 5. Fit Main Models

        # A. Fit Covariates Model (Shared X) - Done once.
        self.coefs['covariates'] = self._solve_shared(cov, y)

        for net in ['positive', 'negative', 'both']:
            # Helper: Reshape inputs into Batches [Perms, Samples, Features]
            def to_batch(t):
                if t.dim() == 2: return t.t().unsqueeze(2)  # [N, P] -> [P, N, 1]
                return t.permute(1, 0, 2)  # [N, P, F] -> [P, N, F]

            X_conn = to_batch(feats[net]['conn'])
            X_resid = to_batch(feats[net]['resid'])

            # For Full Model: Expand covariates to match batch size
            cov_expanded = cov.unsqueeze(0).expand(n_perms, -1, -1)
            X_full = torch.cat([X_conn, cov_expanded], dim=2)

            # Target y: [N, P] -> [P, N, 1]
            y_batch = y.t().unsqueeze(2)

            # Fit Batched Models (Using Fast Cholesky)
            self.coefs[f'connectome_{net}'] = self._solve_batched_fast(X_conn, y_batch)
            self.coefs[f'residuals_{net}'] = self._solve_batched_fast(X_resid, y_batch)
            self.coefs[f'full_{net}'] = self._solve_batched_fast(X_full, y_batch)

        return self

    def predict(self, X, covariates):
        """
        Predicts y for all permutations.
        Returns tensor with shape [N_samples, N_models, N_networks, N_runs].
        """
        X = torch.as_tensor(X, device=self.device, dtype=torch.float32)
        cov = torch.as_tensor(covariates, device=self.device, dtype=torch.float32)

        n_samples = X.size(0)
        n_perms = self.edges.size(2)  # Shape: [Features, Networks, Runs]

        # Initialize output tensor
        predictions = torch.zeros(n_samples, len(Models), len(Networks), n_perms,
                                  device=self.device, dtype=torch.float32)

        # Calculate network strengths and residuals
        # X: [N_samples, N_features] @ edges: [N_features, N_runs] -> [N_samples, N_runs]
        pos_str = X @ self.edges[:, Networks.positive, :].float()
        neg_str = X @ self.edges[:, Networks.negative, :].float()
        pos_resid = pos_str - self._pred_shared(cov, self.resid_models['pos'])
        neg_resid = neg_str - self._pred_shared(cov, self.resid_models['neg'])

        # Covariates model (same for all networks)
        predictions[:, Models.covariates, :, :] = self._pred_shared(cov, self.coefs['covariates']).unsqueeze(1)

        # Helper to convert strengths to batch format [N_perms, N_samples, N_features]
        def to_batch(strength_tensor):
            if strength_tensor.dim() == 2:  # [N_samples, N_perms] -> [N_perms, N_samples, 1]
                return strength_tensor.t().unsqueeze(2)
            else:  # [N_samples, N_perms, 2] -> [N_perms, N_samples, 2]
                return strength_tensor.permute(1, 0, 2)

        # Process each network
        networks = [
            (Networks.positive, pos_str, pos_resid),
            (Networks.negative, neg_str, neg_resid),
            (Networks.both, torch.stack([pos_str, neg_str], dim=2), torch.stack([pos_resid, neg_resid], dim=2))
        ]

        cov_expanded = cov.unsqueeze(0).expand(n_perms, -1, -1)

        for net_idx, conn_strength, resid_strength in networks:
            X_conn = to_batch(conn_strength)
            X_resid = to_batch(resid_strength)
            X_full = torch.cat([X_conn, cov_expanded], dim=2)

            predictions[:, Models.connectome, net_idx, :] = self._pred_batched(X_conn, self.coefs[f'connectome_{net_idx.name}']).squeeze(2).t()
            predictions[:, Models.residuals, net_idx, :] = self._pred_batched(X_resid, self.coefs[f'residuals_{net_idx.name}']).squeeze(2).t()
            predictions[:, Models.full, net_idx, :] = self._pred_batched(X_full, self.coefs[f'full_{net_idx.name}']).squeeze(2).t()

        return predictions

    # --- SOLVERS ---

    def _solve_shared(self, X, y):
        """
        Multi-Target Solver (Shared X).
        X: [N, F], y: [N, P] -> Beta: [F+1, P]
        """
        ones = torch.ones(X.size(0), 1, device=self.device, dtype=X.dtype)
        X_design = torch.cat([ones, X], dim=1)

        XtX = X_design.T @ X_design
        XtX.diagonal().add_(1e-8)
        Xty = X_design.T @ y

        return torch.linalg.solve(XtX, Xty)

    def _pred_shared(self, X, beta):
        ones = torch.ones(X.size(0), 1, device=self.device, dtype=X.dtype)
        X_design = torch.cat([ones, X], dim=1)
        return X_design @ beta

    def _solve_batched_fast(self, X, y):
        """
        Fast Batched Solver (Unique X per perm).
        X: [P, N, F], y: [P, N, 1] -> Beta: [P, F+1, 1]
        """
        P, N, F = X.shape
        ones = torch.ones(P, N, 1, device=self.device, dtype=X.dtype)
        X_design = torch.cat([ones, X], dim=2)  # [P, N, F+1]

        XtX = torch.bmm(X_design.transpose(1, 2), X_design)
        XtX.diagonal(dim1=-2, dim2=-1).add_(1e-8)
        Xty = torch.bmm(X_design.transpose(1, 2), y)

        return torch.linalg.solve(XtX, Xty)

    def _pred_batched(self, X, beta):
        P, N, F = X.shape
        ones = torch.ones(P, N, 1, device=self.device, dtype=X.dtype)
        X_design = torch.cat([ones, X], dim=2)
        return torch.bmm(X_design, beta)

    def get_network_strengths(self, X: np.ndarray, covariates: np.ndarray):
        """
        Calculates network strengths for ALL permutations simultaneously.
        """
        # 1. Convert Inputs to Tensors
        X_tensor = torch.as_tensor(X, device=self.device, dtype=torch.float32)
        # FIX: Convert covariates to Tensor here
        cov_tensor = torch.as_tensor(covariates, device=self.device, dtype=torch.float32)

        # 2. Vectorized Strength Calculation
        # X: [N_samples, N_features], edges: [N_features, 2, N_runs]
        # Result: [N_samples, 2, N_runs]
        all_strengths = torch.einsum('nf,frp->nrp', X_tensor, self.edges.float())

        # 3. Separate Positive and Negative Strengths [N_samples, N_runs]
        pos_str = all_strengths[:, Networks.positive, :]
        neg_str = all_strengths[:, Networks.negative, :]

        # 4. Calculate Predictions from Covariates
        # Note: We pass cov_tensor (Tensor) instead of covariates (NumPy)
        pred_pos = self._pred_shared(cov_tensor, self.resid_models['pos'])
        pred_neg = self._pred_shared(cov_tensor, self.resid_models['neg'])

        # --- IMPORTANT LOGIC CORRECTION ---
        # Your original code attempted to view these as (-1, 1).
        # However, your residual models were trained on [N, P] targets in fit(),
        # so they produce [N, P] predictions. Reshaping to (-1, 1) will break
        # the subtraction if N_perms > 1.

        # 5. Calculate Residuals
        # Direct subtraction works because shapes match: [N, P] - [N, P]
        pos_resid = pos_str - pred_pos
        neg_resid = neg_str - pred_neg

        return {
            "connectome": {
                Networks.positive.name: pos_str,
                Networks.negative.name: neg_str
            },
            "residuals": {
                Networks.positive.name: pos_resid,
                Networks.negative.name: neg_resid
            }
        }