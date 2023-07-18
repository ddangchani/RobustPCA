# Matrix percentile-thresholding

# Computes the percentile thresholding matrix for the matrix X

import numpy as np

def threshold(X, gamma):
    """
    gamma : Thresholding percentile
    """
    X_out = np.copy(X)

    nrow, ncol = X.shape

    X_abs = np.abs(X)
    row_percentiles = np.percentile(X_abs, (1-gamma)*100, axis=1)
    col_percentiles = np.percentile(X_abs, (1-gamma)*100, axis=0)
    
    row_indices, col_indices = np.where(
        (X_abs > row_percentiles[:, np.newaxis]) & (X_abs > col_percentiles)
    )
    X_out[row_indices, col_indices] = 0

    return X_out