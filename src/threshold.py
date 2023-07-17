# Matrix Hard-thresholding

# Computes the hard-thresholding of a given matrix X

import numpy as np

def threshold(X, gamma, sparsity):
    """
    gamma : Thresholding percentile
    sparsity : Sparsity level
    """
    n1, n2 = X.shape
    t1 = np.ones(n1)
    t2 = np.ones(n2)
    X = X * sparsity

    for i in range(n1):
        tt = np.sort(np.abs(X[i,:]))[::-1]  # Sort in descending order
        t1[i] = tt[np.floor(gamma * np.sum(sparsity[i, :])).astype(int) + 1]

    for j in range(n2):
        tt = np.sort(np.abs(X[:,j]))[::-1]
        t2[j] = tt[np.floor(gamma * np.sum(sparsity[:, j])).astype(int) + 1]

    threshold1 = np.tile(t1, (n2, 1)).T
    threshold2 = np.tile(t2, (n1, 1))

    # sum two threshold matrices
    threshold = threshold1 + threshold2

    # Hard-thresholding
    X = np.multiply(X, threshold >= 1)

    return X