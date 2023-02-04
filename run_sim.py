import numpy as np
import matplotlib.pyplot as plt
import EpidemicModel

beta,eta,sigma,gamma,delta = 1,.1,.1,.1,.5

Theta_A = np.array([[0, 0,      0,   delta],
                    [0, 0,      0,     0],
                    [0, eta,    0,     0],
                    [0, gamma, sigma,  0]])

B_A = np.array([[0,     0,  0,  0],
                [beta,  0,  0,  0],
                [0,     0,  0,  0],
                [0,     0,  0,  0]])


O_cal, I_cal, I_max = [0,2,3], [1], 0.1

Y_cal = [[], [(1,0)], [], []]
Z_cal = [[(1,0)], [], [], []]

lambdas = (Theta_A > 0).astype(float)
# lambdas[2,1] = 0.7
omegas = (B_A > 0).astype(float)

model = EpidemicModel(Theta_A, B_A, lambdas, omegas, I_cal, O_cal, Y_cal, Z_cal, I_max)

x0 = np.array([.99,0.01,0,0])
numsteps = 1000
dt = .1

control = False

xs, uv = model.simulate(numsteps, dt, x0, control)

xs_arr = np.array(xs)
labels = ['S','I','Q','R']
plt.figure()
for i in range(len(Theta_A)):
    plt.subplot(2,2,i+1)
    plt.plot(xs_arr[:,i])
    plt.grid()
    plt.title(labels[i])
plt.tight_layout()

if control:
    uv_arr = np.array(uv)
    labels = ['$u_{IQ}$','$u_{IR}$','$v_{SI}$']
    plt.figure(figsize=(4,3))
    plt.plot(uv_arr)
    plt.grid()
    plt.legend(labels)

plt.show()