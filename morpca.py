# Manifold Optimization for Robust PCA

import numpy as np
import scipy
from src.threshold import threshold
from src.projective_retraction import projective_retraction
from src.orthographic_retraction import orthographic_retraction

def morpca(Y, r, gamma, sparsity,
           retraction = ["projective","orthographic"],
           maxiter = 100,
           stepsize = 0.1,
           verbose = False):
    """
    Morpca: Manifold Optimization for Robust PCA
    Y: Input data matrix
    r: Rank of low-rank component
    gamma: Value between 0 and 1 corresponding to percentile for the hard thresholding procedure
    retraction : Retraction to use for the manifold optimization
    verbose : If true, prints the objective value at each iteration
    """

    # Set up data structure
    L_list = np.array([])
    gradient_list = np.array([])
    objective = np.array([])
    n1, n2 = Y.shape

    # Initialize
    L = threshold(Y, gamma, sparsity)
    SVD = np.linalg.svd(np.dot(L.T, L))
    U = SVD[0]

    L_list = np.append(L_list, U[:,0:r] @ U[:,0:r].T @ L)
    gradient_list = np.append(gradient_list, L)
    objective = np.append(objective, np.linalg.norm(L, ord = 'fro')) # Frobenius norm

    if verbose:
        print("Iteration 0: Objective value = ", objective[0])

    # Iterate
    for i in range(maxiter):

        if retraction == "projective" or retraction == "p":
            
            L, gradient = projective_retraction(L, Y, r, gamma, stepsize, sparsity)

        elif retraction == "orthographic" or retraction == "o":

            L, gradient = orthographic_retraction(L, Y, r, gamma, stepsize, sparsity)

        else:
            raise ValueError("Invalid retraction : Argument must be either 'projective' or 'orthographic'")
            break

        L_list = np.append(L_list, L)
        gradient_list = np.append(gradient_list, gradient)
        objective = np.append(objective, np.linalg.norm(L, ord = 'fro')) # Frobenius norm

        if verbose:
            print("Iteration ", i+1, ": Objective value = ", objective[i+1])

    return L_list, gradient_list, objective
