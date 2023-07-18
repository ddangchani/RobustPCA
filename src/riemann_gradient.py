import numpy as np

def riemann_gradient(L, D):

    U, S, Vt = np.linalg.svd(L, full_matrices=False)
    V = Vt.T

    UUt = np.dot(U, U.T)
    VVt = np.dot(V, Vt)

    # Compute the gradient
    gradient = UUt @ D + D @ VVt - UUt @ (D @ VVt)

    return gradient