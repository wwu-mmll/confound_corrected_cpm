import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


class TorchLinearRegression(nn.Module):
    def __init__(self, input_dim=None):
        self.beta = None

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        X = X.float()
        y = y.float().view(-1, 1)
        # Closed-form OLS: (X^T X)^-1 X^T y
        XtX = torch.matmul(X.T, X)
        Xty = torch.matmul(X.T, y)
        self.beta = torch.linalg.solve(XtX, Xty)  # [d,1]
        return self

    def predict(self, X: torch.Tensor):
        X = X.float()
        return (torch.matmul(X, self.beta)).squeeze()


class LinearCPMModel:
    def __init__(self, device='cuda'):
        self.models = ModelDict()
        self.models_residuals = {}
        self.device = torch.device(device)

        self.output_keys = [
            "connectome_pos", "connectome_neg", "connectome_both",
            "covariates",
            "residual_pos", "residual_neg", "residual_both",
            "full_pos", "full_neg", "full_both"
        ]


    def fit(self, X, y, covariates, edge_mask: torch.Tensor):
        X = X.to(self.device)
        y = y.to(self.device)
        covariates = covariates.to(self.device)
        mask = edge_mask.to(self.device).to(dtype=X.dtype)

        connectome = {}
        residuals = {}

        conn_pos = (X * mask).sum(dim=1, keepdim=True)  # [n,1]
        conn_neg = (X * (1 - mask)).sum(dim=1, keepdim=True)  # [n,1]

        connectome["positive"] = conn_pos
        connectome["negative"] = conn_neg

        for net in ["positive", "negative"]:
            reg = TorchLinearRegression(covariates.size(1)).fit(  # .to(self.device)
                covariates, connectome[net].squeeze()
            )
            self.models_residuals[net] = reg
            preds = reg.predict(covariates).unsqueeze(1)
            residuals[net] = connectome[net] - preds

        residuals["both"] = torch.cat([residuals["positive"], residuals["negative"]], dim=1)
        connectome["both"] = torch.cat([connectome["positive"], connectome["negative"]], dim=1)

        for net in ["positive", "negative", "both"]:
            self.models["connectome"][net] = TorchLinearRegression(connectome[net].size(1)).fit(  # .to(self.device)
                connectome[net], y
            )
            self.models["covariates"][net] = TorchLinearRegression(covariates.size(1)).fit(
                covariates, y
            )
            self.models["residuals"][net] = TorchLinearRegression(residuals[net].size(1)).fit(
                residuals[net], y
            )
            full_input = torch.cat([connectome[net], covariates], dim=1)
            self.models["full"][net] = TorchLinearRegression(full_input.size(1)).fit(full_input, y)

        return self


    def predict(self, X: torch.Tensor, covariates: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        """
        Predict all models. Returns tensor [n, n_outputs].
        """
        X, covariates, edge_mask = X.to(self.device), covariates.to(self.device), edge_mask.to(self.device)

        conn_pos = X[:, edge_mask].sum(dim=1, keepdim=True)
        conn_neg = X[:, ~edge_mask].sum(dim=1, keepdim=True)
        conn_both = torch.cat([conn_pos, conn_neg], dim=1)

        preds_pos_resid = self.models_residuals["positive"].predict(covariates).unsqueeze(1)
        preds_neg_resid = self.models_residuals["negative"].predict(covariates).unsqueeze(1)
        resid_pos = conn_pos - preds_pos_resid
        resid_neg = conn_neg - preds_neg_resid
        resid_both = torch.cat([resid_pos, resid_neg], dim=1)

        outputs = {
            "connectome": {
                "positive": self.models["connectome"]["positive"].predict(conn_pos),
                "negative": self.models["connectome"]["negative"].predict(conn_neg),
                "both": self.models["connectome"]["both"].predict(conn_both),
            },
            "covariates": {
                "positive": self.models["covariates"]["positive"].predict(covariates),
                "negative": self.models["covariates"]["negative"].predict(covariates),
                "both": self.models["covariates"]["both"].predict(covariates),
            },
            "residuals": {
                "positive": self.models["residuals"]["positive"].predict(resid_pos),
                "negative": self.models["residuals"]["negative"].predict(resid_neg),
                "both": self.models["residuals"]["both"].predict(resid_both),
            },
            "full": {
                "positive": self.models["full"]["positive"].predict(torch.cat([conn_pos, covariates], dim=1)),
                "negative": self.models["full"]["negative"].predict(torch.cat([conn_neg, covariates], dim=1)),
                "both": self.models["full"]["both"].predict(torch.cat([conn_both, covariates], dim=1)),
            }
        }
        return outputs


    def get_network_strengths(self, X: torch.Tensor, covariates: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        """
        Return [n, 4]: [conn_pos, conn_neg, resid_pos, resid_neg].
        """
        X, covariates, edge_mask = X.to(self.device), covariates.to(self.device), edge_mask.to(self.device)

        conn_pos = X[:, edge_mask].sum(dim=1, keepdim=True)
        conn_neg = X[:, ~edge_mask].sum(dim=1, keepdim=True)

        preds_pos = self.models_residuals["positive"].predict(covariates).unsqueeze(1)
        preds_neg = self.models_residuals["negative"].predict(covariates).unsqueeze(1)

        resid_pos = conn_pos - preds_pos
        resid_neg = conn_neg - preds_neg

        stacked = torch.cat([conn_pos, conn_neg, resid_pos, resid_neg], dim=1)  # [n, 4]
        out = {
            "connectome": {
                "positive": conn_pos,
                "negative": conn_neg,
            },
            "residuals": {
                "positive": resid_pos,
                "negative": resid_neg,
            },
        }

        return out
