import numpy as np
import torch

from cccpm.constants import Networks, Models, TaskType


class LinearCPM:
    """
    A PyTorch implementation of CPM optimized for speed.

    Supports both regression and binary classification tasks.
    For classification, uses logistic regression (linear model + sigmoid).

    Optimizations:
    1. Vectorized Over Permutations: Fits all N_perms models in parallel.
    2. Fast Cholesky Solver: Uses Normal Equations (XtX^-1 Xty) instead of SVD/QR.
    3. Shared Covariate Logic: Handles fixed covariates efficiently.

    Input Shapes:
      - X: [N_samples, N_features]
      - y: [N_samples, N_perms]
    """
    name = "LinearCPM"

    def __init__(self, edges, device='cpu', task_type=TaskType.regression):
        """
        Args:
            edges: Boolean masks tensor with shape [N_features, 2, N_runs].
                   Dimension 1 corresponds to [Positive, Negative].
            device: 'cpu' or 'cuda'
            task_type: TaskType.regression or TaskType.classification
        """
        self.device = torch.device(device)
        self.task_type = task_type

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
        # Select solvers: logistic (IRLS) for classification, OLS for regression
        if self.task_type == TaskType.classification:
            solve_shared = self._solve_shared_logistic
            solve_batched = self._solve_batched_logistic
        else:
            solve_shared = self._solve_shared
            solve_batched = self._solve_batched_fast

        # A. Fit Covariates Model (Shared X) - Done once.
        self.coefs['covariates'] = solve_shared(cov, y)

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

            # Fit Models
            self.coefs[f'connectome_{net}'] = solve_batched(X_conn, y_batch)
            self.coefs[f'residuals_{net}'] = solve_batched(X_resid, y_batch)
            self.coefs[f'full_{net}'] = solve_batched(X_full, y_batch)

        return self

    def predict(self, X, covariates, return_proba=False):
        """
        Predicts y for all permutations.

        Args:
            X: Features [N_samples, N_features]
            covariates: Covariates [N_samples, N_cov]
            return_proba: If True and task is classification, returns probabilities.
                         If False and task is classification, returns class predictions (0/1).
                         Ignored for regression.

        Returns:
            Tensor with shape [N_samples, N_models, N_networks, N_runs].
            - For regression: continuous predictions
            - For classification with return_proba=True: probabilities [0, 1]
            - For classification with return_proba=False: class labels {0, 1}
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

        # Apply activation for classification
        if self.task_type == TaskType.classification:
            predictions = torch.sigmoid(predictions)

            # Convert probabilities to class labels if requested
            if not return_proba:
                predictions = (predictions > 0.5).float()

        return predictions

    def predict_proba(self, X, covariates):
        """
        Predict class probabilities for classification tasks.

        Args:
            X: Features [N_samples, N_features]
            covariates: Covariates [N_samples, N_cov]

        Returns:
            Tensor with shape [N_samples, N_models, N_networks, N_runs] containing probabilities.

        Note:
            Only applicable for classification tasks. For regression, returns same as predict().
        """
        if self.task_type == TaskType.classification:
            return self.predict(X, covariates, return_proba=True)
        else:
            return self.predict(X, covariates)

    def predict_class(self, X, covariates):
        """
        Predict class labels (0/1) for classification tasks.

        Args:
            X: Features [N_samples, N_features]
            covariates: Covariates [N_samples, N_cov]

        Returns:
            Tensor with shape [N_samples, N_models, N_networks, N_runs] containing class labels {0, 1}.

        Note:
            Only applicable for classification tasks. For regression, returns same as predict().
        """
        if self.task_type == TaskType.classification:
            return self.predict(X, covariates, return_proba=False)
        else:
            return self.predict(X, covariates)

    # --- Batched-over-params-and-folds fit/predict ---
    #
    # `fit()`/`predict()` above are left untouched and remain the code path
    # used whenever B_params == B_folds == 1 (today's exact per-fold,
    # per-param loop). `fit_batched()`/`predict_batched()` below are used
    # only when folds and/or params are actually batched (self.edges is 5D
    # instead of 3D): they solve the SAME linear models via the SAME
    # `_solve_batched_fast`/`_solve_batched_logistic`/`_pred_batched`
    # helpers, generalised with extra broadcastable batch dims and an
    # explicit `row_mask` so zero-padded rows (see utils.build_fold_batch)
    # never corrupt the fit.

    def fit_batched(self, X, y, covariates, valid_mask=None):
        """
        Batched-over-params-and-folds version of `fit()`.

        Requires `self.edges` shape [N_features, 2, B_params, B_folds, N_perms]
        (5D -- use `fit()` for the plain 3D [N_features, 2, N_perms] case).
        Mathematically identical to calling `fit()` once per (param, fold)
        pair with that fold's real (unpadded) rows and that param's edge
        mask; zero-padding is kept inert via the row_mask discipline
        documented on `_solve_batched_fast`.

        Args:
            X: [B_folds, N, N_features] -- padded fold-batch (utils.build_fold_batch).
            y: [B_folds, N, N_perms] -- padded fold-batch.
            covariates: [B_folds, N, N_cov] -- padded fold-batch.
            valid_mask: optional [B_folds, N] bool; True on real (non-pad)
                        rows. Defaults to all-True (no padding).
        """
        X = torch.as_tensor(X, device=self.device, dtype=torch.float32)
        y = torch.as_tensor(y, device=self.device, dtype=torch.float32)
        cov = torch.as_tensor(covariates, device=self.device, dtype=torch.float32)

        B_folds, N, n_cov = cov.shape
        n_features, _, B_params, edges_folds, n_perms = self.edges.shape
        if edges_folds != B_folds or n_perms != y.shape[-1]:
            raise ValueError(
                f"self.edges folds/perms ({edges_folds}, {n_perms}) must match "
                f"X/y's folds/perms ({B_folds}, {y.shape[-1]})")
        P, B, R = B_params, B_folds, n_perms

        if valid_mask is None:
            valid_mask = torch.ones(B_folds, N, dtype=torch.bool, device=self.device)
        row_mask = valid_mask.to(X.dtype)  # [B, N]

        # 1. Network strengths: combine per-fold X with per-(param,fold,perm)
        #    edges via einsum -> [P, B, R, N].
        edges_pos = self.edges[:, Networks.positive].float()  # [F, P, B, R]
        edges_neg = self.edges[:, Networks.negative].float()
        pos_str = torch.einsum('bnf,fpbr->pbrn', X, edges_pos)
        neg_str = torch.einsum('bnf,fpbr->pbrn', X, edges_neg)

        row_mask_pbrn = row_mask.view(1, B, 1, N)
        pos_str = pos_str * row_mask_pbrn
        neg_str = neg_str * row_mask_pbrn

        cov_b = cov.view(1, B, 1, N, n_cov).expand(P, B, R, N, n_cov)
        row_mask_b = row_mask.view(1, B, 1, N).expand(P, B, R, N)

        solve_batched = (self._solve_batched_logistic if self.task_type == TaskType.classification
                         else self._solve_batched_fast)

        # 2. Residualisers (strength ~ covariates), batched over (P, B, R).
        pos_str_col = pos_str.unsqueeze(-1)  # [P, B, R, N, 1]
        neg_str_col = neg_str.unsqueeze(-1)
        self.resid_models['pos'] = self._solve_batched_fast(cov_b, pos_str_col, row_mask=row_mask_b)
        self.resid_models['neg'] = self._solve_batched_fast(cov_b, neg_str_col, row_mask=row_mask_b)

        pos_resid = pos_str - self._pred_batched(cov_b, self.resid_models['pos'], row_mask=row_mask_b).squeeze(-1)
        neg_resid = neg_str - self._pred_batched(cov_b, self.resid_models['neg'], row_mask=row_mask_b).squeeze(-1)
        pos_resid = pos_resid * row_mask_pbrn
        neg_resid = neg_resid * row_mask_pbrn

        feats = {
            'positive': {'conn': pos_str, 'resid': pos_resid},
            'negative': {'conn': neg_str, 'resid': neg_resid},
            'both': {'conn': torch.stack([pos_str, neg_str], dim=-1),    # [P, B, R, N, 2]
                     'resid': torch.stack([pos_resid, neg_resid], dim=-1)}
        }

        # 3. Covariates-only model (y ~ covariates). Genuinely param-independent
        #    (no edges involved), so it is solved once per (fold, perm) and the
        #    result is broadcast across params rather than recomputed P times.
        y_fr = y.permute(0, 2, 1)                                  # [B, R, N]
        cov_fr = cov.unsqueeze(1).expand(B, R, N, n_cov)
        row_mask_fr = row_mask.view(B, 1, N).expand(B, R, N)
        covariates_beta = solve_batched(cov_fr, y_fr.unsqueeze(-1), row_mask=row_mask_fr)  # [B, R, C+1, 1]
        self.coefs['covariates'] = covariates_beta.unsqueeze(0).expand(P, B, R, n_cov + 1, 1)

        y_batch = y_fr.unsqueeze(0).expand(P, B, R, N).unsqueeze(-1)  # [P, B, R, N, 1]

        for net in ['positive', 'negative', 'both']:
            X_conn = feats[net]['conn']
            X_resid = feats[net]['resid']
            if X_conn.dim() == 4:  # [P,B,R,N] -> [P,B,R,N,1]
                X_conn = X_conn.unsqueeze(-1)
                X_resid = X_resid.unsqueeze(-1)
            X_full = torch.cat([X_conn, cov_b], dim=-1)

            self.coefs[f'connectome_{net}'] = solve_batched(X_conn, y_batch, row_mask=row_mask_b)
            self.coefs[f'residuals_{net}'] = solve_batched(X_resid, y_batch, row_mask=row_mask_b)
            self.coefs[f'full_{net}'] = solve_batched(X_full, y_batch, row_mask=row_mask_b)

        return self

    def predict_batched(self, X, covariates, valid_mask=None, return_proba=False):
        """
        Batched-over-params-and-folds version of `predict()`. See `fit_batched`.

        Args:
            X: [B_folds, N, N_features] -- padded fold-batch (test rows).
            covariates: [B_folds, N, N_cov] -- padded fold-batch.
            valid_mask: optional [B_folds, N] bool. Padded rows still get a
                        (meaningless) prediction value -- callers must ignore
                        them using the same valid_mask, exactly as they must
                        already ignore padded rows in scoring/storage.
            return_proba: as in `predict()`.

        Returns:
            Tensor [N, N_models, N_networks, B_params, B_folds, N_perms].
        """
        X = torch.as_tensor(X, device=self.device, dtype=torch.float32)
        cov = torch.as_tensor(covariates, device=self.device, dtype=torch.float32)

        B_folds, N, n_cov = cov.shape
        n_features, _, B_params, edges_folds, n_perms = self.edges.shape
        P, B, R = B_params, B_folds, n_perms

        if valid_mask is None:
            valid_mask = torch.ones(B_folds, N, dtype=torch.bool, device=self.device)
        row_mask = valid_mask.to(X.dtype)

        predictions = torch.zeros(N, len(Models), len(Networks), P, B, R,
                                  device=self.device, dtype=torch.float32)

        edges_pos = self.edges[:, Networks.positive].float()
        edges_neg = self.edges[:, Networks.negative].float()
        pos_str = torch.einsum('bnf,fpbr->pbrn', X, edges_pos)
        neg_str = torch.einsum('bnf,fpbr->pbrn', X, edges_neg)
        row_mask_pbrn = row_mask.view(1, B, 1, N)
        pos_str = pos_str * row_mask_pbrn
        neg_str = neg_str * row_mask_pbrn

        cov_b = cov.view(1, B, 1, N, n_cov).expand(P, B, R, N, n_cov)
        row_mask_b = row_mask.view(1, B, 1, N).expand(P, B, R, N)

        pos_resid = pos_str - self._pred_batched(cov_b, self.resid_models['pos'], row_mask=row_mask_b).squeeze(-1)
        neg_resid = neg_str - self._pred_batched(cov_b, self.resid_models['neg'], row_mask=row_mask_b).squeeze(-1)
        pos_resid = pos_resid * row_mask_pbrn
        neg_resid = neg_resid * row_mask_pbrn

        # Covariates model: param-independent, computed once (see fit_batched) and broadcast.
        cov_fr = cov.unsqueeze(1).expand(B, R, N, n_cov)
        row_mask_fr = row_mask.view(B, 1, N).expand(B, R, N)
        covariates_pred = self._pred_batched(cov_fr, self.coefs['covariates'][0], row_mask=row_mask_fr)  # [B,R,N,1]
        covariates_pred = covariates_pred.squeeze(-1).permute(2, 0, 1)  # [N, B, R]
        predictions[:, Models.covariates, :, :, :, :] = (
            covariates_pred.view(N, 1, 1, B, R).expand(N, len(Networks), P, B, R))

        networks = [
            (Networks.positive, pos_str, pos_resid),
            (Networks.negative, neg_str, neg_resid),
            (Networks.both, torch.stack([pos_str, neg_str], dim=-1),
             torch.stack([pos_resid, neg_resid], dim=-1)),
        ]

        for net_idx, conn_strength, resid_strength in networks:
            X_conn = conn_strength if conn_strength.dim() == 5 else conn_strength.unsqueeze(-1)
            X_resid = resid_strength if resid_strength.dim() == 5 else resid_strength.unsqueeze(-1)
            X_full = torch.cat([X_conn, cov_b], dim=-1)

            pred_conn = self._pred_batched(X_conn, self.coefs[f'connectome_{net_idx.name}'], row_mask=row_mask_b)
            pred_resid = self._pred_batched(X_resid, self.coefs[f'residuals_{net_idx.name}'], row_mask=row_mask_b)
            pred_full = self._pred_batched(X_full, self.coefs[f'full_{net_idx.name}'], row_mask=row_mask_b)

            # pred_*: [P, B, R, N, 1] -> [N, P, B, R]
            predictions[:, Models.connectome, net_idx, :, :, :] = pred_conn.squeeze(-1).permute(3, 0, 1, 2)
            predictions[:, Models.residuals, net_idx, :, :, :] = pred_resid.squeeze(-1).permute(3, 0, 1, 2)
            predictions[:, Models.full, net_idx, :, :, :] = pred_full.squeeze(-1).permute(3, 0, 1, 2)

        if self.task_type == TaskType.classification:
            predictions = torch.sigmoid(predictions)
            if not return_proba:
                predictions = (predictions > 0.5).float()

        return predictions

    # --- SOLVERS ---

    # --- Logistic Regression (IRLS) ---

    def _solve_shared_logistic(self, X, y, max_iter=25, tol=1e-6):
        """
        Logistic regression via IRLS for shared X, multiple targets.
        X: [N, F], y: [N, P] -> Beta: [F+1, P]

        Since IRLS weights differ per permutation, this internally expands
        to batched operations over permutations.
        """
        N = X.size(0)
        P = y.size(1)
        ones = torch.ones(N, 1, device=self.device, dtype=X.dtype)
        X_design = torch.cat([ones, X], dim=1)  # [N, F+1]

        # Expand X for batched operations: [P, N, F+1]
        X_batch = X_design.unsqueeze(0).expand(P, -1, -1)
        # y: [N, P] -> [P, N, 1]
        y_batch = y.t().unsqueeze(2)

        # Initialize with OLS solution, reshaped to [P, F+1, 1]
        beta = self._solve_shared(X, y).t().unsqueeze(2)

        for _ in range(max_iter):
            logits = torch.bmm(X_batch, beta)  # [P, N, 1]
            p = torch.sigmoid(logits)

            W = (p * (1 - p)).clamp(min=1e-8)  # [P, N, 1]

            # Working response
            z = logits + (y_batch - p) / W  # [P, N, 1]

            # Weighted least squares
            sqrt_W = torch.sqrt(W)
            X_w = X_batch * sqrt_W  # [P, N, F+1]
            z_w = z * sqrt_W  # [P, N, 1]

            XtX = torch.bmm(X_w.transpose(1, 2), X_w)
            XtX.diagonal(dim1=-2, dim2=-1).add_(1e-8)
            Xtz = torch.bmm(X_w.transpose(1, 2), z_w)

            beta_new = torch.linalg.solve(XtX, Xtz)

            if torch.max(torch.abs(beta_new - beta)) < tol:
                beta = beta_new
                break
            beta = beta_new

        # Reshape back to [F+1, P]
        return beta.squeeze(2).t()

    def _solve_batched_logistic(self, X, y, row_mask=None, max_iter=25, tol=1e-6):
        """
        Logistic regression via IRLS for batched inputs.
        X: [..., N, F], y: [..., N, 1] -> Beta: [..., F+1, 1]
        `...` may be zero or more broadcastable leading batch dims (just
        perms, as in `fit()`, or params/folds/perms when batching those too
        via `fit_batched()`).

        row_mask: optional [..., N, 1] (broadcastable), 1 on real rows, 0 on
                  padded rows (see `_solve_batched_fast`). Zeroing padded
                  rows via X's own zero-padding is NOT sufficient here
                  (sigmoid(0) = 0.5 != 0, so a padded row would otherwise get
                  non-zero IRLS weight) -- padded rows are explicitly masked
                  out of the weights every iteration instead.
        """
        if row_mask is None:
            ones = torch.ones(*X.shape[:-1], 1, device=self.device, dtype=X.dtype)
            mask = ones
        else:
            mask = row_mask.to(X.dtype)
            if mask.dim() < X.dim():
                mask = mask.unsqueeze(-1)
            mask = mask.expand(*X.shape[:-1], 1)
            ones = mask
        X_design = torch.cat([ones, X], dim=-1)  # [..., N, F+1]

        # Initialize with OLS solution
        beta = self._solve_batched_fast(X, y, row_mask=row_mask)  # [..., F+1, 1]

        for _ in range(max_iter):
            logits = torch.matmul(X_design, beta)  # [..., N, 1]
            p = torch.sigmoid(logits)

            W = (p * (1 - p)).clamp(min=1e-8) * mask  # [..., N, 1], 0 on padded rows

            z = logits + (y - p) / W.clamp(min=1e-8)  # [..., N, 1]

            sqrt_W = torch.sqrt(W)
            X_w = X_design * sqrt_W  # [..., N, F+1]
            z_w = z * sqrt_W  # [..., N, 1]

            XtX = torch.matmul(X_w.transpose(-1, -2), X_w)
            XtX.diagonal(dim1=-2, dim2=-1).add_(1e-8)
            Xtz = torch.matmul(X_w.transpose(-1, -2), z_w)

            beta_new = torch.linalg.solve(XtX, Xtz)

            if torch.max(torch.abs(beta_new - beta)) < tol:
                beta = beta_new
                break
            beta = beta_new

        return beta

    # --- Linear Regression (OLS) ---

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

    def _solve_batched_fast(self, X, y, row_mask=None):
        """
        Fast Batched Solver (unique X per batch element).
        X: [..., N, F], y: [..., N, 1] -> Beta: [..., F+1, 1]
        `...` may be zero or more broadcastable leading batch dims (just
        perms, as in `fit()`, or params/folds/perms when batching those too
        via `fit_batched()`).

        row_mask: optional [..., N] or [..., N, 1] (broadcastable to X's
                  batch+sample shape), 1 on real rows, 0 on padded rows.
                  Used AS the intercept column instead of a blanket 1, so a
                  padded row's design-matrix row is exactly zero and
                  contributes nothing to X^T X / X^T y (see
                  edge_selection.get_residuals_batched for why zero-row
                  padding is exact for this normal-equations OLS). `X`'s own
                  feature columns must already be zero on padded rows too
                  (the caller is responsible for that, e.g. via masked
                  strength/einsum computation).
        """
        if row_mask is None:
            ones = torch.ones(*X.shape[:-1], 1, device=self.device, dtype=X.dtype)
        else:
            ones = row_mask.to(X.dtype)
            if ones.dim() < X.dim():
                ones = ones.unsqueeze(-1)
            ones = ones.expand(*X.shape[:-1], 1)
        X_design = torch.cat([ones, X], dim=-1)  # [..., N, F+1]

        XtX = torch.matmul(X_design.transpose(-1, -2), X_design)
        XtX.diagonal(dim1=-2, dim2=-1).add_(1e-8)
        Xty = torch.matmul(X_design.transpose(-1, -2), y)

        return torch.linalg.solve(XtX, Xty)

    def _pred_batched(self, X, beta, row_mask=None):
        """X: [..., N, F], beta: [..., F+1, 1] -> [..., N, 1]. See `_solve_batched_fast`."""
        if row_mask is None:
            ones = torch.ones(*X.shape[:-1], 1, device=self.device, dtype=X.dtype)
        else:
            ones = row_mask.to(X.dtype)
            if ones.dim() < X.dim():
                ones = ones.unsqueeze(-1)
            ones = ones.expand(*X.shape[:-1], 1)
        X_design = torch.cat([ones, X], dim=-1)
        return torch.matmul(X_design, beta)

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