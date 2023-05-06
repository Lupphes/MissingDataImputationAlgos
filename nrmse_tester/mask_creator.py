#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Authors: Aude Sportisse with the help of Marine Le Morvan and Boris Muzellec
#Online https://rmisstastic.netlify.app/how-to/python/generate_html/how%20to%20generate%20missing%20values

#Shortened and edited by Adam Michalik


import torch
import numpy as np

from scipy import optimize

import matplotlib.pyplot as plt

def MAR_mask(X, p, p_obs):
    """
    Missing at random mechanism with a logistic masking model. First, a subset of variables with *no* missing values is
    randomly selected. The remaining variables have missing values according to a logistic model with random weights,
    re-scaled so as to attain the desired proportion of missing values on those variables.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated. If a numpy array is provided,
        it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    p_obs : float
        Proportion of variables with *no* missing values that will be used for the logistic masking model.

    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_obs = max(int(p_obs * d), 1) ## number of variables that will have no missing values (at least one variable)
    d_na = d - d_obs ## number of variables that will have missing values

    ### Sample variables that will all be observed, and those with missing values:
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    ### Other variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = _pick_coeffs(X, idxs_obs, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = _fit_intercepts(X[:, idxs_obs], coeffs, p)

    ps = torch.sigmoid(X[:, idxs_obs].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    return mask


def MNAR_mask_logistic(X, p, p_params =.3, exclude_inputs=True):
    """
    Missing not at random mechanism with a logistic masking model. It implements two mechanisms:
    (i) Missing probabilities are selected with a logistic model, taking all variables as inputs. Hence, values that are
    inputs can also be missing.
    (ii) Variables are split into a set of intputs for a logistic model, and a set whose missing probabilities are
    determined by the logistic model. Then inputs are then masked MCAR (hence, missing values from the second set will
    depend on masked values.
    In either case, weights are random and the intercept is selected to attain the desired proportion of missing values.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    p_params : float
        Proportion of variables that will be used for the logistic masking model (only if exclude_inputs).

    exclude_inputs : boolean, default=True
        True: mechanism (ii) is used, False: (i)

    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_params = max(int(p_params * d), 1) if exclude_inputs else d ## number of variables used as inputs (at least 1)
    d_na = d - d_params if exclude_inputs else d ## number of variables masked with the logistic model

    ### Sample variables that will be parameters for the logistic regression:
    idxs_params = np.random.choice(d, d_params, replace=False) if exclude_inputs else np.arange(d)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_params]) if exclude_inputs else np.arange(d)

    ### Other variables will have NA proportions selected by a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = _pick_coeffs(X, idxs_params, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = _fit_intercepts(X[:, idxs_params], coeffs, p)

    ps = torch.sigmoid(X[:, idxs_params].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    ## If the inputs of the logistic model are excluded from MNAR missingness,
    ## mask some values used in the logistic model at random.
    ## This makes the missingness of other variables potentially dependent on masked values

    if exclude_inputs:
        mask[:, idxs_params] = torch.rand(n, d_params) < p

    return mask


def _pick_coeffs(X, idxs_obs=None, idxs_nas=None, self_mask=False):
    """
    Compute the coefficients for logistic regression.

    params:
        X: tensor of shape (n, d) - the input data
        idxs_obs: list or tensor of length d_obs - the indices of the observed features (optional)
        idxs_nas: list or tensor of length d_na - the indices of the features with missing values (optional)
        self_mask: boolean - whether to use a self mask for computing the coefficients (optional)

    return:
        coeffs: tensor of shape (d_obs, d_na) or (d,) - the coefficients for logistic regression
    """
    n, d = X.shape

    # If self_mask is True, generate a random coefficient vector and normalize it
    # based on the standard deviation of the weighted inputs
    if self_mask:
        coeffs = torch.randn(d)
        Wx = X * coeffs
        coeffs /= torch.std(Wx, 0)

    # Otherwise, initialize a random coefficient matrix for the observed inputs
    # and the missing inputs, and normalize based on the standard deviation of the
    # weighted inputs for the observed inputs
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = torch.randn(d_obs, d_na)
        Wx = X[:, idxs_obs].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True)

    # Return the coefficient vector/matrix
    return coeffs


def _fit_intercepts(X, coeffs, p, self_mask=False):
    """
    Compute the coefficients for logistic regression.

    params:
        X: tensor of shape (n, d) - the input data
        idxs_obs: list or tensor of length d_obs - the indices of the observed features (optional)
        idxs_nas: list or tensor of length d_na - the indices of the features with missing values (optional)
        self_mask: boolean - whether to use a self mask for computing the coefficients (optional)

    return:
        coeffs: tensor of shape (d_obs, d_na) or (d,) - the coefficients for logistic regression
    """
        
    # If self_mask is True, initialize an intercept vector of zeros
    # with length equal to the number of coefficients
    if self_mask:
        d = len(coeffs)
        intercepts = torch.zeros(d)
        
        # For each coefficient, define a function that takes an intercept x as input,
        # calculates the mean of the sigmoid activation of the weighted inputs plus x,
        # and returns the difference between this mean and the target sparsity level p
        for j in range(d):
            def f(x):
                return torch.sigmoid(X * coeffs[j] + x).mean().item() - p
            
            # Use the bisection method to find the root of the function f within the range [-50, 50]
            intercepts[j] = optimize.bisect(f, -50, 50)
    # Otherwise, initialize an intercept vector of zeros with length equal to the number of missing inputs
    else:
        d_obs, d_na = coeffs.shape
        intercepts = torch.zeros(d_na)
        
        # For each missing input, define a function that takes an intercept x as input,
        # calculates the mean of the sigmoid activation of the weighted observed inputs plus x,
        # and returns the difference between this mean and the target sparsity level p
        for j in range(d_na):
            def f(x):
                return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p
            
            # Use the bisection method to find the root of the function f within the range [-50, 50]
            intercepts[j] = optimize.bisect(f, -50, 50)
    
    # Return the intercept vector
    return intercepts

