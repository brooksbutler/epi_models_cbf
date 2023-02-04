import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as sopt

import util

class EpidemicModel():
    def __init__(self, Theta, B, lambdas, omegas, I_cal, O_cal, Y_cal, Z_cal, I_max):
        self.Theta = Theta
        self.B =  B

        lam_f, omg_f = util.advance_flatten(lambdas), util.advance_flatten(omegas) 
        self.lam_f = lam_f[(lam_f['val'] > 0) & (lam_f['d1'].isin(I_cal))]
        self.omg_f = omg_f[omg_f['val'] > 0]
        self.c = np.concatenate((self.lam_f['val'].values, self.omg_f['val'].values))

        self.n = len(Theta)
        self.I_max = I_max

        self.I_cal = set(I_cal)
        self.O_cal = set(O_cal)
        self.Y_cal = [set(y) for y in Y_cal]
        self.Z_cal = [set(z) for z in Z_cal]

        self.Theta_L = -util.Laplacian(Theta)
        self.B_L = self.build_B(B)
        

    def simulate(self, numsteps, dt, x0, control=False):
        x = x0
        xs = [x]
        if not control:
            U_L, V_L = np.zeros(self.Theta_L.shape), np.zeros(self.B_L.shape) 
            for _ in range(numsteps):
                x = x + dt*self.x_dot(x, self.Theta_L, self.B_L, U_L, V_L)
                xs.append(x)
            return xs, None
        
        else:
            controls = []
            for _ in range(numsteps):
                U_L, V_L, control_vec = self.solve_UV(x)
                controls.append(control_vec)
                x = x + dt*self.x_dot(x, self.Theta_L, self.B_L, U_L, V_L)
                xs.append(x)
            return xs, controls

    def x_dot(self, x, Theta, B, U_l, V_l):
        x_d = ((Theta + U_l) + np.kron(x.T,np.eye(len(x)))@(B-V_l))@x
        return x_d
    
    def build_B(self, B_A):
        B = np.zeros((self.n**2,self.n))

        for i, (Y_i, Z_i) in enumerate(zip(self.Y_cal,self.Z_cal)):
            if Y_i:
                for y in Y_i:
                    j, k = y
                    B[self.n*k+i, j] = B_A[j,k]
            
            if Z_i:
                for z in Z_i:
                    j, k = z
                    B[self.n*k+i, j] = -B_A[j,k]
        return B

    def h(self, x):
        inds = [i for i in self.I_cal]
        return self.I_max - np.sum(x[inds])

    def alpha(self,x):
        return x**2

    def d(self, x):
        inds = [i for i in self.O_cal]
        d = (self.Theta_L  + np.kron(x.T,np.eye(len(x)))@self.B_L)@x
        return np.sum(d[inds])

    def solve_UV(self, x):
        # Assemble upper bounds for LP solver
        a_l = x[self.lam_f['d1'].values]
        a_o = np.array([x[i]*x[j] for _, i, j in self.omg_f.itertuples(index=False)])

        # Find solution us LP
        A = -np.atleast_2d(np.concatenate((a_l,a_o)))
        b = self.d(x) + self.alpha(self.h(x))
        
        sol = sopt.linprog(self.c,A,b)
        sol_vals = sol.x 
        
        # Reconstruct weights
        U, V = np.zeros((self.n,self.n)), np.zeros((self.n,self.n))
        for k, tup in enumerate(self.lam_f.itertuples(index=False)):
            _, i, j = tup
            U[i,j] = sol_vals[k]

        for k, tup in enumerate(self.omg_f.itertuples(index=False)):
            _, i, j = tup
            V[i,j] = sol_vals[len(self.lam_f)+k]
        
        # Return controller in proper format
        return  -util.Laplacian(U), self.build_B(V), sol_vals


