# Projective retraction 

import numpy as np
import scipy
from src.threshold import threshold
from src.approximation import rank_r_approximation

def projective_retraction(L, Y, r, gamma, eta, sparsity):
    """
    Projective retraction
    L: approximation L
    Y: Input data matrix
    r: Rank of low-rank component
    gamma: Value between 0 and 1 corresponding to percentile for the hard thresholding procedure
    eta: Step size
    sparsity: Sparsity of sparse component
    """
    
    # Gradient Descent
    gradient = threshold(L-Y, gamma, sparsity)
    L_tmp = L - eta * L.T @ gradient
    L_out = rank_r_approximation(L_tmp, r)

    return L_out, gradient