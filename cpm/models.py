import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from pygam import LinearGAM, s
from functools import reduce
import operator


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
    def __init__(self, edges):
        """
        Initialize the LinearCPMModel.

        Parameters
        ----------
        edges : dict
            Dictionary containing indices of edges for 'positive' and 'negative' networks.
        """
        self.models = ModelDict()
        self.models_residuals = {}
        self.edges = edges

    def fit(self, X, y, covariates):
        """
        Fit the CPM model.

        This method fits multiple linear regression models for the connectome, covariates,
        residuals, and full model using the provided data.

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
        """
        connectome = {}
        residuals = {}
        for network in ['positive', 'negative']:
            # Compute sum_positive and sum_negative
            connectome[network] = np.sum(X[:, self.edges[network]], axis=1).reshape(-1, 1)
            self.models_residuals[network] =  LinearRegression().fit(covariates, connectome[network])
            residuals[network] = connectome[network] - self.models_residuals[network].predict(covariates)

        residuals['both'] = np.hstack((residuals['positive'], residuals['negative']))
        connectome['both'] = np.hstack((connectome['positive'], connectome['negative']))

        for network in NetworkDict().keys():
            self.models['connectome'][network] = LinearRegression().fit(connectome[network], y)
            self.models['covariates'][network] = LinearRegression().fit(covariates, y)
            self.models['residuals'][network] = LinearRegression().fit(residuals[network], y)
            self.models['full'][network] = LinearRegression().fit(np.hstack((connectome[network], covariates)), y)

        return self

    def predict(self, X, covariates):
        """
        Predict using the fitted CPM model.

        This method generates predictions for the target variable using the
        connectome, covariates, residuals, and full models.

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
        connectome = {}
        residuals = {}
        for network in ['positive', 'negative']:
            # Compute sum_positive and sum_negative
            connectome[network] = np.sum(X[:, self.edges[network]], axis=1).reshape(-1, 1)
            self.models_residuals[network] =  LinearRegression().fit(covariates, connectome[network])
            residuals[network] = connectome[network] - self.models_residuals[network].predict(covariates)

        residuals['both'] = np.hstack((residuals['positive'], residuals['negative']))
        connectome['both'] = np.hstack((connectome['positive'], connectome['negative']))

        predictions = ModelDict()
        for network in ['positive', 'negative', 'both']:
            predictions['connectome'][network] = self.models['connectome'][network].predict(connectome[network])
            predictions['covariates'][network] = self.models['covariates'][network].predict(covariates)
            predictions['residuals'][network] = self.models['residuals'][network].predict(residuals[network])
            predictions['full'][network] = self.models['full'][network].predict(np.hstack((connectome[network], covariates)))

        return predictions

    def get_network_strengths(self, X, covariates):
        connectome = {}
        residuals = {}
        for network in ['positive', 'negative']:
            # Compute sum_positive and sum_negative
            connectome[network] = np.sum(X[:, self.edges[network]], axis=1).reshape(-1, 1)
            residuals[network] = connectome[network] - self.models_residuals[network].predict(covariates)

        return {"connectome": connectome, "residuals": residuals}


class GAMCPMModel(LinearCPMModel):

    def __init__(self, edges, gam_params):
        super().__init__(edges)
        self.gam_params = gam_params

    def _make_gam_terms(self, n_features):
        
        gam_terms = []
        for i in range(n_features):
            gam_terms.append(s(i, n_splines = self.gam_params["n_splines"]))

        if len(gam_terms) == 1:
            return gam_terms[0]
        else:
            return reduce(operator.add, gam_terms)
    
    def fit(self, X, y, covariates):
        connectome = {}
        residuals = {}
        for network in ['positive', 'negative']:
            if len(self.edges[network]) > 0:
                connectome[network] = X[:, self.edges[network]]
            else:
                connectome[network] = np.zeros((X.shape[0], 1))  
            self.models_residuals[network] =  LinearRegression().fit(covariates, connectome[network])
            residuals[network] = connectome[network] - self.models_residuals[network].predict(covariates)


        residuals['both'] = np.hstack((residuals['positive'], residuals['negative']))
        connectome['both'] = np.hstack((connectome['positive'], connectome['negative']))

        for network in NetworkDict().keys():
            self.models['connectome'][network] = LinearGAM(self._make_gam_terms(connectome[network].shape[1])).fit(connectome[network], y)
            self.models['covariates'][network] = LinearGAM(self._make_gam_terms(covariates.shape[1])).fit(covariates, y)
            self.models['residuals'][network] = LinearGAM(self._make_gam_terms(residuals[network].shape[1])).fit(residuals[network], y)
            self.models['full'][network] = LinearGAM(self._make_gam_terms(connectome[network].shape[1] + covariates.shape[1])).fit(np.hstack((connectome[network], covariates)), y)

        return self
    
    def predict(self, X, covariates):
        """
        Predict using the fitted CPM model.

        This method generates predictions for the target variable using the
        connectome, covariates, residuals, and full models.

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
        connectome = {}
        residuals = {}
        for network in ['positive', 'negative']:
            if len(self.edges[network]) > 0:
                connectome[network] = X[:, self.edges[network]]
            else:
                connectome[network] = np.zeros((X.shape[0], 1))  
            self.models_residuals[network] =  LinearRegression().fit(covariates, connectome[network])#.mean(axis=1, keepdims=True))
            residuals[network] = connectome[network] - self.models_residuals[network].predict(covariates)

        residuals['both'] = np.hstack((residuals['positive'], residuals['negative']))
        connectome['both'] = np.hstack((connectome['positive'], connectome['negative']))

        predictions = ModelDict()
        for network in ['positive', 'negative', 'both']:
            predictions['connectome'][network] = self.models['connectome'][network].predict(connectome[network])
            predictions['covariates'][network] = self.models['covariates'][network].predict(covariates)
            predictions['residuals'][network] = self.models['residuals'][network].predict(residuals[network])
            predictions['full'][network] = self.models['full'][network].predict(np.hstack((connectome[network], covariates)))

        return predictions

    def get_network_strengths(self, X, covariates):
        connectome = {}
        residuals = {}
        for network in ['positive', 'negative']:
            # Compute sum_positive and sum_negative
            connectome[network] = np.sum(X[:, self.edges[network]], axis=1).reshape(-1, 1)
            residuals[network] = connectome[network] - self.models_residuals[network].predict(covariates)

        return {"connectome": connectome, "residuals": residuals}


class DecisionTreeCPMModel(LinearCPMModel):

    def __init__(self, edges, tree_params = None):
        super().__init__(edges)
        self.gam_params = tree_params if tree_params else {}
    
    def fit(self, X, y, covariates):
        connectome = {}
        residuals = {}
        for network in ['positive', 'negative']:
            if len(self.edges[network]) > 0:
                connectome[network] = X[:, self.edges[network]]
            else:
                connectome[network] = np.zeros((X.shape[0], 1))  
            self.models_residuals[network] =  LinearRegression().fit(covariates, connectome[network])#.mean(axis=1, keepdims=True))
            residuals[network] = connectome[network] - self.models_residuals[network].predict(covariates)

        residuals['both'] = np.hstack((residuals['positive'], residuals['negative']))
        connectome['both'] = np.hstack((connectome['positive'], connectome['negative']))

        for network in NetworkDict().keys():
            self.models['connectome'][network] = DecisionTreeRegressor().fit(connectome[network], y)
            self.models['covariates'][network] = DecisionTreeRegressor().fit(covariates, y)
            self.models['residuals'][network] = DecisionTreeRegressor().fit(residuals[network], y)
            self.models['full'][network] = DecisionTreeRegressor().fit(np.hstack((connectome[network], covariates)), y)

        return self
    
    def predict(self, X, covariates):
        """
        Predict using the fitted CPM model.

        This method generates predictions for the target variable using the
        connectome, covariates, residuals, and full models.

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
        connectome = {}
        residuals = {}
        for network in ['positive', 'negative']:
            if len(self.edges[network]) > 0:
                connectome[network] = X[:, self.edges[network]]
            else:
                connectome[network] = np.zeros((X.shape[0], 1))  
            self.models_residuals[network] =  LinearRegression().fit(covariates, connectome[network].mean(axis=1, keepdims=True))
            residuals[network] = connectome[network] - self.models_residuals[network].predict(covariates)

        residuals['both'] = np.hstack((residuals['positive'], residuals['negative']))
        connectome['both'] = np.hstack((connectome['positive'], connectome['negative']))

        predictions = ModelDict()
        for network in ['positive', 'negative', 'both']:
            predictions['connectome'][network] = self.models['connectome'][network].predict(connectome[network])
            predictions['covariates'][network] = self.models['covariates'][network].predict(covariates)
            predictions['residuals'][network] = self.models['residuals'][network].predict(residuals[network])
            predictions['full'][network] = self.models['full'][network].predict(np.hstack((connectome[network], covariates)))

        return predictions

    def get_network_strengths(self, X, covariates):
        connectome = {}
        residuals = {}
        for network in ['positive', 'negative']:
            # Compute sum_positive and sum_negative
            connectome[network] = np.sum(X[:, self.edges[network]], axis=1).reshape(-1, 1)
            residuals[network] = connectome[network] - self.models_residuals[network].predict(covariates)

        return {"connectome": connectome, "residuals": residuals}


class NonLinearCPMModel(LinearCPMModel):

    def __init__(self, edges, model, params = None):
        super().__init__(edges)
        self.model = model
        self.params = params if params else {}

    def _make_gam_terms(self, n_features):
    
        gam_terms = []
        for i in range(n_features):
            gam_terms.append(s(i, n_splines = self.params["n_splines"]))

        if len(gam_terms) == 1:
            return gam_terms[0]
        else:
            return reduce(operator.add, gam_terms)
        

    def get_cpm_model(self, connectome_shape=None):

        if self.model == 'GAMCPM':
            return LinearGAM(self._make_gam_terms(connectome_shape), lam=self.params['lam'])
        elif self.model == 'DecisionTreeCPM':
            return DecisionTreeRegressor(random_state=0, **self.params)

    
    def fit(self, X, y, covariates):
        connectome = {}
        residuals = {}
        for network in ['positive', 'negative']:
            if len(self.edges[network]) > 0:
                connectome[network] = X[:, self.edges[network]]
            else:
                connectome[network] = np.zeros((X.shape[0], 1))  
            self.models_residuals[network] =  LinearRegression().fit(covariates, connectome[network])
            residuals[network] = connectome[network] - self.models_residuals[network].predict(covariates)

        residuals['both'] = np.hstack((residuals['positive'], residuals['negative']))
        connectome['both'] = np.hstack((connectome['positive'], connectome['negative']))

        for network in NetworkDict().keys():
            self.models['connectome'][network] = self.get_cpm_model(connectome[network].shape[1]).fit(connectome[network], y)
            self.models['covariates'][network] = self.get_cpm_model(covariates.shape[1]).fit(covariates, y)
            self.models['residuals'][network] = self.get_cpm_model(residuals[network].shape[1]).fit(residuals[network], y)
            self.models['full'][network] = self.get_cpm_model(connectome[network].shape[1] + covariates.shape[1]).fit(np.hstack((connectome[network], covariates)), y)

        return self
    
    def predict(self, X, covariates):
        """
        Predict using the fitted CPM model.

        This method generates predictions for the target variable using the
        connectome, covariates, residuals, and full models.

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
        connectome = {}
        residuals = {}
        for network in ['positive', 'negative']:
            if len(self.edges[network]) > 0:
                connectome[network] = X[:, self.edges[network]]
            else:
                connectome[network] = np.zeros((X.shape[0], 1))  
            self.models_residuals[network] =  LinearRegression().fit(covariates, connectome[network].mean(axis=1, keepdims=True))
            residuals[network] = connectome[network] - self.models_residuals[network].predict(covariates)

        residuals['both'] = np.hstack((residuals['positive'], residuals['negative']))
        connectome['both'] = np.hstack((connectome['positive'], connectome['negative']))

        predictions = ModelDict()
        for network in ['positive', 'negative', 'both']:
            predictions['connectome'][network] = self.models['connectome'][network].predict(connectome[network])
            predictions['covariates'][network] = self.models['covariates'][network].predict(covariates)
            predictions['residuals'][network] = self.models['residuals'][network].predict(residuals[network])
            predictions['full'][network] = self.models['full'][network].predict(np.hstack((connectome[network], covariates)))

        return predictions

    def get_network_strengths(self, X, covariates):
        connectome = {}
        residuals = {}
        for network in ['positive', 'negative']:
            # Compute sum_positive and sum_negative
            connectome[network] = np.sum(X[:, self.edges[network]], axis=1).reshape(-1, 1)
            residuals[network] = connectome[network] - self.models_residuals[network].predict(covariates)

        return {"connectome": connectome, "residuals": residuals}



