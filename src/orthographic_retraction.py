# Orthographic retraction

import numpy as np
from src.threshold import threshold

def orthographic_retraction(L, Y, r, gamma, eta, sparsity):
    """
    Orthographic retraction
    L: Current estimate of low-rank component
    Y: Input data matrix
    r: Rank of low-rank component
    gamma: Value between 0 and 1 corresponding to percentile for the hard thresholding procedure
    eta: Step size
    sparsity: Sparsity parameter
    """
    
    gradient = threshold(L-Y, gamma, sparsity)

    Q = L[:, np.random.choice(L.shape[1], r, replace=False)] # Randomly choose r columns of L
    R = L[np.random.choice(L.shape[0], r, replace=False), :] # Randomly choose r rows of L

    # Gradient Descent
    L_tmp = L - eta * gradient
    QtL_tmp = Q.T @ L_tmp
    L_out = np.dot(L_tmp, R) @ np.linalg.inv(np.dot(QtL_tmp, R)) @ QtL_tmp

    return L_out, gradient