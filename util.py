import numpy as np
import pandas as pd

def Laplacian(A):
    L = np.zeros(A.shape)
    n = len(A)
    for i in range(n):
        for j in range(n):
            if i==j:
                L[i,j] = np.sum(A[:,j])
            else:
                L[i,j] = -A[i,j]
    return L

def advance_flatten(M):
    indices = np.array(list(np.ndindex(M.shape)))
    df = pd.DataFrame({'val': M.flatten(), 'd0': indices[:, 0], 'd1': indices[:, 1]})
    return df

