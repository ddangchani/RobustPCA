# Manifold Optimization for Robust PCA

import numpy as np
from src.threshold import threshold
from src.projective_retraction import projective_retraction
from src.orthographic_retraction import orthographic_retraction


def morpca(Y, r, gamma,
           retraction = ["projective","orthographic"],
           maxiter = 100,
           stepsize = 0.1,
           verbose = False):
    """
    Morpca: Manifold Optimization for Robust PCA
    Y: Input data matrix
    r: Rank of low-rank component
    gamma: Value between 0 and 1 corresponding to percentile for the hard thresholding procedure(near 1 for strong thresholding)
    retraction : Retraction to use for the manifold optimization
    verbose : If true, prints the objective value at each iteration
    """

    # Set up data structure
    L_list = []
    objective = []

    # Initialize
    L = threshold(Y, gamma)
    SVD = np.linalg.svd(np.dot(L, L.T))
    U = SVD[0]

    L_list.append(U[:,0:r] @ U[:,0:r].T @ L)
    objective.append(np.linalg.norm(L, ord = 'fro')) # Frobenius norm

    if verbose:
        print("Iteration 0: Objective value = ", objective[0])

    # Iterate
    for i in range(maxiter):

        if retraction == "projective" or retraction == "p":
            
            L, _ = projective_retraction(L_list[i], Y, r, gamma, stepsize)

        elif retraction == "orthographic" or retraction == "o":

            L, _ = orthographic_retraction(L_list[i], Y, r, gamma, stepsize)

        else:
            raise ValueError("Invalid retraction : Argument must be either 'projective' or 'orthographic'")

        L_list.append(L)
        objective.append(np.linalg.norm(L, ord = 'fro')) # Frobenius norm

        if verbose:
            print(f"Iteration {i+1}: Objective value = ", objective[i+1])

    return {"Y" : Y,
            "rank" : r,
            "gamma" : gamma,
            "retraction" : retraction,
            "maxiter" : maxiter,
            "stepsize" : stepsize,
            "solution" : L_list,
            "objective" : objective}
