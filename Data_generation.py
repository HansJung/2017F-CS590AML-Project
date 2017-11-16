import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

def inv_logit(z):
    return np.exp(z) / (1+np.exp(z))


np.random.seed(100)

c1 = -1; c1_ = 1; c2 = -1; c2_ = 1
theta1 = 0.5; theta2 = 0.3; theta3 = 0.7
sigma = 10
N = 10000; Ns = 50
U1_ = np.random.binomial(1,theta1,N)
U2_ = np.random.binomial(1,theta2,N)
U3_ = np.random.binomial(1,theta3,N)

U1 = []; U2 = []; X = []; X_intv = []; Y = []; Y_intv = []; Z = []


for idx in range(N):
    u1_ = U1_[idx]
    u2_ = U2_[idx]
    u3_ = U3_[idx]
    mu1 = c1 * u1_ + c1_*(1-u1_)
    mu2 = c2 * u2_ + c2_*(1-u2_)

    u1 = np.random.normal(loc=mu1, scale=sigma, size=1)[0]
    u2 = np.random.normal(loc=mu2, scale=sigma, size=1)[0]

    z = u1+u2
    x = round((inv_logit(z) + u1_ + u3_)/3)
    if idx < N/2:
        x_intv = 1
    else:
        x_intv = 0
    y = round((inv_logit(z) + x + u2_ + u3_)/4)
    y_intv = round((inv_logit(z) + x_intv + u2_ + u3_)/4)

    U1.append(u1); U2.append(u2); X.append(x)
    X_intv.append(x_intv), Y.append(y); Y_intv.append(y_intv); Z.append(z)

X_obs = np.asarray(X); Y_obs = np.asarray(Y); Z_obs = np.asarray(Z)
Obs = pd.DataFrame({'Z':Z_obs,'X':X_obs,'Y':Y_obs})

X_intv = np.asarray(X_intv); Y_intv = np.asarray(Y_intv)
Intv_L = pd.DataFrame({'X':X_intv,'Y':Y_intv, 'Z':Z_obs})

sample_indices = np.random.choice(N,Ns,replace=False)
X_sintv = X_intv[sample_indices]; Y_sintv = Y_intv[sample_indices]; Z_sintv = Z_obs[sample_indices]

Intv_S = pd.DataFrame({'X':X_sintv, 'Y':Y_sintv, 'Z':Z_sintv})