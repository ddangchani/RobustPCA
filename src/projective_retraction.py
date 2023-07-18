# Projective retraction 

import numpy as np
from src.threshold import threshold
from src.approximation import rank_r_approximation
from src.riemann_gradient import riemann_gradient


def projective_retraction(L, Y, r, gamma, eta):
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
    gradient = threshold(L-Y, gamma)
    L_tmp = L - eta * riemann_gradient(L, gradient)
    L_out = rank_r_approximation(L_tmp, r)

    return L_out, gradient