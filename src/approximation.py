# Rank r approximation

import numpy as np

def rank_r_approximation(A, r):
    U, S, V = np.linalg.svd(A, full_matrices=False)
    U_r = U[:, :r]
    S_r = np.diag(S[:r])
    V_r = V[:r, :]
    return np.dot(np.dot(U_r, S_r), V_r)