import numpy as np


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

def x_dot(x,Theta, B, U_l, V_l):
    x_d = ((Theta + U_l) + np.kron(x.T,np.eye(len(x)))@(B-V_l))@x
    return x_d 



