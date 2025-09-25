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
    # we can choose here, either use linear nn with adam optimizer or use closed form OLS (faster)

    # def __init__(self, input_dim):
    #     super().__init__()
    #     self.linear = nn.Linear(input_dim, 1, bias=True)
    #
    # def fit(self, X, y, lr=1e-2, epochs=500):  # time 80%
    #     X = X.float()
    #     y = y.float()
    #
    #     optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    #     loss_fn = nn.MSELoss()
    #
    #     for _ in range(epochs):
    #         optimizer.zero_grad()
    #         preds = self.linear(X)
    #         loss = loss_fn(preds.squeeze(), y)
    #         loss.backward()
    #         optimizer.step()
    #     return self
    #
    # def predict(self, X):
    #     X = X.float()
    #     with torch.no_grad():
    #         return self.linear(X).squeeze()


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

        # connectome = {}
        # residuals = {}
        #
        # for net in ['positive', 'negative']:
        #     edge_idx = self.edges[net]
        #     connectome[net] = X[:, edge_idx].sum(dim=1, keepdim=True)
        #     reg = TorchLinearRegression(covariates.size(1)).to(self.device).fit(covariates, connectome[net].squeeze())
        #     self.models_residuals[net] = reg
        #     preds = reg.predict(covariates).unsqueeze(1)
        #     residuals[net] = connectome[net] - preds
        #
        # residuals['both'] = torch.cat([residuals['positive'], residuals['negative']], dim=1)
        # connectome['both'] = torch.cat([connectome['positive'], connectome['negative']], dim=1)
        #
        # for net in ['positive', 'negative', 'both']:
        #     # Connectome-only model
        #     self.models['connectome'][net] = TorchLinearRegression(connectome[net].size(1)).to(self.device).fit(
        #         connectome[net], y)
        #     # Covariate-only model
        #     self.models['covariates'][net] = TorchLinearRegression(covariates.size(1)).to(self.device).fit(covariates,
        #                                                                                                    y)
        #     # Residual model
        #     self.models['residuals'][net] = TorchLinearRegression(residuals[net].size(1)).to(self.device).fit(
        #         residuals[net], y)
        #     # Full model
        #     full_input = torch.cat([connectome[net], covariates], dim=1)
        #     self.models['full'][net] = TorchLinearRegression(full_input.size(1)).to(self.device).fit(full_input, y)
        #
        # return self

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

        # outputs = [
        #     self.models["connectome"]["positive"].predict(conn_pos),
        #     self.models["connectome"]["negative"].predict(conn_neg),
        #     self.models["connectome"]["both"].predict(conn_both),
        #     self.models["covariates"]["positive"].predict(covariates),
        #     self.models["covariates"]["negative"].predict(covariates),
        #     self.models["covariates"]["both"].predict(covariates),
        #     self.models["residuals"]["positive"].predict(resid_pos),
        #     self.models["residuals"]["negative"].predict(resid_neg),
        #     self.models["residuals"]["both"].predict(resid_both),
        #     self.models["full"]["positive"].predict(torch.cat([conn_pos, covariates], dim=1)),
        #     self.models["full"]["negative"].predict(torch.cat([conn_neg, covariates], dim=1)),
        #     self.models["full"]["both"].predict(torch.cat([conn_both, covariates], dim=1)),
        # ]
        # stack = torch.stack(outputs, dim=1)  # [n, 12]
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

    # def predict(self, X, covariates):
    #     X = X.to(self.device)
    #     covariates = covariates.to(self.device)
    #
    #     connectome = {}
    #     residuals = {}
    #
    #     for net in ['positive', 'negative']:
    #         edge_idx = self.edges[net]
    #         connectome[net] = X[:, edge_idx].sum(dim=1, keepdim=True)
    #         preds = self.models_residuals[net].predict(covariates).unsqueeze(1)
    #         residuals[net] = connectome[net] - preds
    #
    #     residuals['both'] = torch.cat([residuals['positive'], residuals['negative']], dim=1)
    #     connectome['both'] = torch.cat([connectome['positive'], connectome['negative']], dim=1)
    #
    #     predictions = ModelDict()
    #     for net in ['positive', 'negative', 'both']:
    #         predictions['connectome'][net] = self.models['connectome'][net].predict(connectome[net])
    #         predictions['covariates'][net] = self.models['covariates'][net].predict(covariates)
    #         predictions['residuals'][net] = self.models['residuals'][net].predict(residuals[net])
    #         full_input = torch.cat([connectome[net], covariates], dim=1)
    #         predictions['full'][net] = self.models['full'][net].predict(full_input)
    #
    #     return predictions

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

    # def get_network_strengths(self, X, covariates):
    #     X = X.to(self.device)
    #     covariates = covariates.to(self.device)
    #
    #     connectome = {}
    #     residuals = {}
    #     for net in ['positive', 'negative']:
    #         edge_idx = self.edges[net]
    #         connectome[net] = X[:, edge_idx].sum(dim=1, keepdim=True)
    #         preds = self.models_residuals[net].predict(covariates).unsqueeze(1)
    #         residuals[net] = connectome[net] - preds
    #
    #     return {"connectome": connectome, "residuals": residuals}

# class LinearCPMModel:
#     """
#     Linear Connectome-based Predictive Modeling (CPM) implementation.
#
#     This class implements a linear CPM model, allowing for fitting and prediction
#     based on connectome data, covariates, and residuals.
#
#     Attributes
#     ----------
#     models : ModelDict
#         A dictionary containing the fitted models for different networks and data types
#         (connectome, covariates, residuals, and full model).
#     models_residuals : dict
#         A dictionary storing linear regression models used to calculate residuals
#         for connectome data, controlling for covariates.
#     edges : dict
#         A dictionary defining the edges (features) used for each network (e.g., 'positive', 'negative').
#
#     Parameters
#     ----------
#     edges : dict
#         Dictionary containing indices of edges for 'positive' and 'negative' networks.
#     """
#     def __init__(self, edges):
#         """
#         Initialize the LinearCPMModel.
#
#         Parameters
#         ----------
#         edges : dict
#             Dictionary containing indices of edges for 'positive' and 'negative' networks.
#         """
#         self.models = ModelDict()
#         self.models_residuals = {}
#         self.edges = edges
#
#     def fit(self, X, y, covariates):
#         """
#         Fit the CPM model.
#
#         This method fits multiple linear regression models for the connectome, covariates,
#         residuals, and full model using the provided data.
#
#         Parameters
#         ----------
#         X : numpy.ndarray
#             A 2D array of shape (n_samples, n_features) representing the connectome data.
#         y : numpy.ndarray
#             A 1D array of shape (n_samples,) representing the target variable.
#         covariates : numpy.ndarray
#             A 2D array of shape (n_samples, n_covariates) representing the covariates.
#
#         Returns
#         -------
#         LinearCPMModel
#             The fitted CPM model instance.
#         """
#         connectome = {}
#         residuals = {}
#         for network in ['positive', 'negative']:
#             # Compute sum_positive and sum_negative
#             connectome[network] = np.sum(X[:, self.edges[network]], axis=1).reshape(-1, 1)
#             self.models_residuals[network] = LinearRegression().fit(covariates, connectome[network])
#             residuals[network] = connectome[network] - self.models_residuals[network].predict(covariates)
#
#         residuals['both'] = np.hstack((residuals['positive'], residuals['negative']))
#         connectome['both'] = np.hstack((connectome['positive'], connectome['negative']))
#
#         for network in NetworkDict().keys():
#             self.models['connectome'][network] = LinearRegression().fit(connectome[network], y)
#             self.models['covariates'][network] = LinearRegression().fit(covariates, y)
#             self.models['residuals'][network] = LinearRegression().fit(residuals[network], y)
#             self.models['full'][network] = LinearRegression().fit(np.hstack((connectome[network], covariates)), y)
#
#         return self
#
#     def predict(self, X, covariates):
#         """
#         Predict using the fitted CPM model.
#
#         This method generates predictions for the target variable using the
#         connectome, covariates, residuals, and full models.
#
#         Parameters
#         ----------
#         X : numpy.ndarray
#             A 2D array of shape (n_samples, n_features) representing the connectome data.
#         covariates : numpy.ndarray
#             A 2D array of shape (n_samples, n_covariates) representing the covariates.
#
#         Returns
#         -------
#         ModelDict
#             A dictionary containing predictions for each network and model type
#             (connectome, covariates, residuals, and full model).
#         """
#         connectome = {}
#         residuals = {}
#         for network in ['positive', 'negative']:
#             # Compute sum_positive and sum_negative
#             connectome[network] = np.sum(X[:, self.edges[network]], axis=1).reshape(-1, 1)
#             residuals[network] = connectome[network] - self.models_residuals[network].predict(covariates)
#
#         residuals['both'] = np.hstack((residuals['positive'], residuals['negative']))
#         connectome['both'] = np.hstack((connectome['positive'], connectome['negative']))
#
#         predictions = ModelDict()
#         for network in ['positive', 'negative', 'both']:
#             predictions['connectome'][network] = self.models['connectome'][network].predict(connectome[network])
#             predictions['covariates'][network] = self.models['covariates'][network].predict(covariates)
#             predictions['residuals'][network] = self.models['residuals'][network].predict(residuals[network])
#             predictions['full'][network] = self.models['full'][network].predict(np.hstack((connectome[network], covariates)))
#
#         return predictions
#
#     def get_network_strengths(self, X, covariates):
#         connectome = {}
#         residuals = {}
#         for network in ['positive', 'negative']:
#             # Compute sum_positive and sum_negative
#             connectome[network] = np.sum(X[:, self.edges[network]], axis=1).reshape(-1, 1)
#             residuals[network] = connectome[network] - self.models_residuals[network].predict(covariates)
#         return {"connectome": connectome, "residuals": residuals}
