import numpy as np
from Model import*

# Define parameters for SIQRS
beta,eta,sigma,gamma,delta = 1,1,1,1,1
Theta_A = np.array([0, 0,      0,   delta],
                   [0, 0,      0,     0],
                   [0, eta,    0,     0],
                   [0, gamma, sigma,  0])

B_A = np.array([0,     0,  0,  0],
               [beta,  0,  0,  0],
               [0,     0,  0,  0],
               [0,     0,  0,  0])

X_cal = set([1])
O_cal = set([0,2,3])

Y_cal = [set([(1,0)]),set(),set(),set()]
Z_cal = [set(),set([(1,0)]),set(),set()]