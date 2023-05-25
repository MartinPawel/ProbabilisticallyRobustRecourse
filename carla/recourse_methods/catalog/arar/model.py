import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

# required for the local linearization of the ANN model
from lime.lime_tabular import LimeTabularExplainer

from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.catalog.arar.arar_recourse import ArArRecourse
from carla.recourse_methods.processing import (
    merge_default_parameters,
)

from carla import log


class ARAR(RecourseMethod):
    """
    Implementation of ARAR

    Parameters
    ----------
    mlmodel : carla.model.MLModel
        Black-Box-Model
    hyperparams : dict
        Dictionary containing hyperparameters. See notes below for its contents.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.
    encode_normalize_order_factuals:
        Uses encoder and scaler from black-box-model to preprocess data as needed.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

    .. [1] Ricardo Dominguez-Olmedo, Amir Hossein-Karimi, and Bernhard Schölkopf. 2022. "On the Adversarial Robustness of Causal Algorithmic Recourse".
           
        Note that this method is a slight variation of Upadhyay et al (2021):  "Towards Robust and Reliable Algorithmic Recourse".
           NeurIPS 2021.
    """
    
    _DEFAULT_HYPERPARAMS = {
        "discretize_continuous": False,
        "sample_around_instance": True,
        "delta": 0.01,
        "lambda_": 0.001
    }
    
    def __init__(
            self,
            mlmodel,
            hyperparams: Dict,
            coeffs: Optional[np.ndarray] = None,
            intercept: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(mlmodel)
        self._data = mlmodel.data
        
        # normalize and encode data
        self._norm_enc_data = self.encode_normalize_order_factuals(
            self._data.raw, with_target=True
        )
        
        # Get hyperparameter
        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )
        
        self._discretize_continuous = checked_hyperparams["discretize_continuous"]
        self._sample_around_instance = checked_hyperparams["sample_around_instance"]
        self._delta = checked_hyperparams["delta"]
        self._coeffs, self._intercept = coeffs, intercept
        self._lambda = checked_hyperparams["lambda_"]
    
    def _get_lime_coefficients(
            self, factual: np.array
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        """
        Their method only works on linear models. To make it work for arbitrary non-linear networks
        we need to do a local linear approximation of the non-linear network for every instance.

        Parameters
        ----------
        factuals : pd.DataFrame
        Instances we want to get lime coefficients

        Returns
        -------
        coeffs : np.ndArray
        intercepts : np.ndArray

        """
        lime_data = self._norm_enc_data[self._mlmodel.feature_input_order]
        explainer = LimeTabularExplainer(training_data=lime_data.values,
                                         discretize_continuous=self._discretize_continuous,
                                         sample_around_instance=self._sample_around_instance,
                                         feature_selection='none')
        
        exp = explainer.explain_instance(factual,
                                         self._mlmodel.raw_model.prob_predict,
                                         num_features=self._norm_enc_data.shape[1])
        
        coefficients = exp.local_exp[1]
        intercept = exp.intercept[1]
        
        # sort from column index 0 to d
        coefficients = sorted(coefficients, key=lambda x: x[0])
        coefs = np.zeros(len(coefficients))
        
        for j in range(len(coefficients)):
            coefs[j] = coefficients[j][1]
        
        return coefs, np.array(intercept).reshape(-1)
    
    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        
        cfs = []
        coeffs = self._coeffs
        intercept = self._intercept
        
        # to keep matching indexes for iterrows and coeffs
        factuals = factuals.reset_index()
        factuals_enc_norm = self.encode_normalize_order_factuals(factuals)
        
        # generate counterfactuals
        for index, row in factuals_enc_norm.iterrows():
            # asserts are essential for mypy typechecking
            # assert coeffs is not None
            # assert intercepts is not None
            factual_enc_norm = row.values
            # coeff = coeffs[index]
            # intercept = intercepts[index]
            
            # Default counterfactual value if no action flips the prediction
            target_shape = factual_enc_norm.shape[0]
            empty = np.empty(target_shape)
            empty[:] = np.nan
            counterfactual = empty
            
            # Check if we need local linear approximation: this is only done for the nonlinear model
            if (coeffs is None) and (intercept is None):
                log.info("Start generating LIME coefficients")
                rec = ArArRecourse(W=coeffs, W0=intercept, feature_costs=None, delta_max=self._delta)
                coeffs_, intercept_ = self._get_lime_coefficients(factual_enc_norm)
                log.info("Finished generating LIME coefficients")
                rec.set_W(coeffs_)
                rec.set_W0(intercept_)
                candidate_cf = rec.get_recourse(factual_enc_norm, lamb=self._lambda)
            
            else:
                # Local explanations via LIME generate coeffs and intercepts per instance, while global explanations
                # via input parameter need to be set into correct shape [num_of_instances, num_of_features]

                rec = ArArRecourse(W=coeffs, W0=intercept, feature_costs=None, delta_max=self._delta)
                candidate_cf = rec.get_recourse(factual_enc_norm, lamb=self._lambda)
            
            pred_cf = np.argmax(self._mlmodel.predict_proba(candidate_cf.reshape((1, -1))))
            pred_f = np.argmax(
                self._mlmodel.predict_proba(factual_enc_norm.reshape((1, -1)))
            )
            
            if pred_cf != pred_f:
                counterfactual = candidate_cf.squeeze()
            
            cfs.append(counterfactual)
        
        # Convert output into correct format
        cfs = np.array(cfs)
        df_cfs = pd.DataFrame(cfs, columns=self._mlmodel.feature_input_order)
        df_cfs[self._mlmodel.data.target] = np.argmax(
            self._mlmodel.predict_proba(cfs), axis=1
        )
        
        return df_cfs

