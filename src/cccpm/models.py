import torch


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


class TorchLinearRegression:
    def __init__(self, device=None):
        self.coef_ = None
        self.intercept_ = None
        self.beta = None
        self.device = device

    def fit(self, X, y):
        device = self.device if self.device is not None else X.device

        X = X.to(torch.float64).to(device)
        y = y.to(torch.float64).to(device)

        ones = torch.ones((X.size(0), 1), dtype=torch.float64, device=device)
        X_design = torch.cat([ones, X], dim=1)

        lambda_reg = 1e-8
        XtX = X_design.T @ X_design
        XtX = XtX + lambda_reg * torch.eye(XtX.size(0), dtype=torch.float64, device=device)
        Xty = X_design.T @ y

        beta = torch.linalg.solve(XtX, Xty)
        #print("fit beta:", beta)
        self.beta = beta

        self.intercept_ = beta[0].item()
        self.coef_ = beta[1:].cpu().numpy()

        return self

    def predict(self, X: torch.Tensor):
        device = self.beta.device

        X = X.to(torch.float64).to(device)

        ones = torch.ones((X.size(0), 1), dtype=torch.float64, device=device)
        X_design = torch.cat([ones, X], dim=1)

        return (X_design @ self.beta)


class LinearCPMModel:
    """
    Linear Connectome-based Predictive Modeling (CPM) implementation.

    This class implements a linear CPM model, allowing for fitting and prediction
    based on connectome data, covariates, and residuals.

    Attributes
    ----------
    models : ModelDict
        A dictionary containing the fitted models for different networks and data types
        (connectome, covariates, residuals, and full model).
    models_residuals : dict
        A dictionary storing linear regression models used to calculate residuals
        for connectome data, controlling for covariates.
    edges : dict
        A dictionary defining the edges (features) used for each network (e.g., 'positive', 'negative').

    Parameters
    ----------
    edges : dict
        Dictionary containing indices of edges for 'positive' and 'negative' networks.
    """
    name = "LinearCPMModel"
    def __init__(self, device='cuda'):
        """
        Initialize the LinearCPMModel.

        Parameters
        ----------
        edges : dict
            Dictionary containing indices of edges for 'positive' and 'negative' networks.
        """
        self.models = ModelDict()
        self.models_residuals = {}
        self.device = torch.device(device)

        self.output_keys = [
            "connectome_pos", "connectome_neg", "connectome_both",
            "covariates",
            "residual_pos", "residual_neg", "residual_both",
            "full_pos", "full_neg", "full_both"
        ]

    def fit(self, X, y, covariates, pos_edges: torch.Tensor, neg_edges: torch.Tensor):
        """
        Fit the CPM model.

        This method fits multiple linear regression models for the connectome, covariates,
        residuals, and full model using the provided data.

        vmap-safe CPM fit.
        X:          [n, P]
        y:          [n]
        covariates: [n, C]
        pos_edges:  [P] boolean mask
        neg_edges:  [P] boolean mask

        Parameters
        ----------
        X : numpy.ndarray
            A 2D array of shape (n_samples, n_features) representing the connectome data.
        y : numpy.ndarray
            A 1D array of shape (n_samples,) representing the target variable.
        covariates : numpy.ndarray
            A 2D array of shape (n_samples, n_covariates) representing the covariates.

        Returns
        -------
        LinearCPMModel
            The fitted CPM model instance.
        :param X:
        :param y:
        :param covariates:
        :param pos_edges:
        :param neg_edges:
        :return:


        """
        X = X.to(self.device)
        y = y.to(self.device)
        covariates = covariates.to(self.device)
        pos_mask = pos_edges.to(self.device).float()
        neg_mask = neg_edges.to(self.device).float()
        connectome = {}
        residuals = {}

        conn_pos = (X * pos_mask).sum(dim=1, keepdim=True)  # [n,1]
        conn_neg = (X * neg_mask).sum(dim=1, keepdim=True)  # [n,1]

        connectome["positive"] = conn_pos
        connectome["negative"] = conn_neg

        for net in ["positive", "negative"]:
            reg = TorchLinearRegression().fit(
                covariates, connectome[net].squeeze(1)
            )
            self.models_residuals[net] = reg

            preds = reg.predict(covariates).unsqueeze(1)  # [n,1]
            residuals[net] = connectome[net] - preds  # [n,1]

        residuals["both"] = torch.cat([residuals["positive"], residuals["negative"]], dim=1)
        connectome["both"] = torch.cat([connectome["positive"], connectome["negative"]], dim=1)

        for net in ["positive", "negative", "both"]:
            # connectome-only model
            self.models["connectome"][net] = TorchLinearRegression().fit(connectome[net], y)

            # covariates-only model
            self.models["covariates"][net] = TorchLinearRegression().fit(covariates, y)

            # residuals-only model
            self.models["residuals"][net] = TorchLinearRegression().fit(residuals[net], y)

            # full model
            full_input = torch.cat([connectome[net], covariates], dim=1)
            self.models["full"][net] = TorchLinearRegression().fit(full_input, y)

        return self

    def predict(self, X: torch.Tensor, covariates: torch.Tensor,
                pos_edges: torch.Tensor, neg_edges: torch.Tensor):
        """
        Predict using the fitted CPM model.

        This method generates predictions for the target variable using the
        connectome, covariates, residuals, and full models.

        vmap-compatible CPM prediction.
        X:          [n, P]
        covariates: [n, C]
        pos_edges:  [P] boolean mask of positive edges
        neg_edges:  [P] boolean mask of negative edges

        Parameters
        ----------
        X : numpy.ndarray
            A 2D array of shape (n_samples, n_features) representing the connectome data.
        covariates : numpy.ndarray
            A 2D array of shape (n_samples, n_covariates) representing the covariates.

        Returns
        -------
        ModelDict
            A dictionary containing predictions for each network and model type
            (connectome, covariates, residuals, and full model).
        """
        X = X.to(self.device)
        covariates = covariates.to(self.device)

        pos_mask = pos_edges.to(self.device).float()  # [P]
        neg_mask = neg_edges.to(self.device).float()  # [P]

        conn_pos = (X * pos_mask).sum(dim=1, keepdim=True)  # [n,1]
        conn_neg = (X * neg_mask).sum(dim=1, keepdim=True)  # [n,1]
        conn_both = torch.cat([conn_pos, conn_neg], dim=1)  # [n,2]

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
                "positive": self.models["full"]["positive"].predict(
                    torch.cat([conn_pos, covariates], dim=1)
                ),
                "negative": self.models["full"]["negative"].predict(
                    torch.cat([conn_neg, covariates], dim=1)
                ),
                "both": self.models["full"]["both"].predict(
                    torch.cat([conn_both, covariates], dim=1)
                ),
            }
        }

        return outputs

    def get_network_strengths(
            self,
            X: torch.Tensor,
            covariates: torch.Tensor,
            pos_edges: torch.Tensor,
            neg_edges: torch.Tensor,
    ):
        """
        GPU version matching CPU behavior.
        Computes:
            connectome['positive'] = sum over pos_edges
            connectome['negative'] = sum over neg_edges
            residuals[...] = connectome[...] - model.predict(covariates)
        X : [n, p] tensor
        covariates : [n, c] tensor
        pos_edges : LongTensor of indices
        neg_edges : LongTensor of indices
        {
          "connectome": { "positive": ..., "negative": ... },
          "residuals": { "positive": ..., "negative": ... }
        }
        """
        X = X.to(self.device)
        covariates = covariates.to(self.device)
        pos_edges = pos_edges.to(self.device)
        neg_edges = neg_edges.to(self.device)

        # --- network strengths ---
        conn_pos = X[:, pos_edges].sum(dim=1, keepdim=True)  # [n,1]
        conn_neg = X[:, neg_edges].sum(dim=1, keepdim=True)  # [n,1]
        # --- covariate regression residuals ---
        preds_pos = self.models_residuals["positive"].predict(covariates).unsqueeze(1)
        preds_neg = self.models_residuals["negative"].predict(covariates).unsqueeze(1)
        resid_pos = conn_pos - preds_pos
        resid_neg = conn_neg - preds_neg

        return {
            "connectome": {
                "positive": conn_pos,
                "negative": conn_neg,
            },
            "residuals": {
                "positive": resid_pos,
                "negative": resid_neg,
            },
        }
