from Data_generation import *
from scipy import stats
from scipy.optimize import minimize


def KL(p_obs, x):
    return p_obs * np.log(p_obs / x) + (1 - p_obs) * np.log((1 - p_obs) / (1 - x))

def entropy(p):
    return -p * np.log(p) - (1 - p) * np.log(1 - p)

def partial_entropy(p,x):
    if x == 1:
        return -p*np.log(p)
    elif x == 0:
        return -(1-p)*np.log(1-p)


def cross_entropy(p,q):
    return -p * np.log(q) - (1 - p) * np.log(1 - q)

def causal_bound(p_obs, p_hat, lboudns, ubounds, Hx, minmax_mode):
    def fun(x):
        if minmax_mode == 'min':
            return x
        elif minmax_mode == 'max':
            return -x

    def fun_deriv(x):
        if minmax_mode == 'min':
            return 1
        elif minmax_mode == 'max':
            return -1

    cons = ({'type': 'ineq',
             'fun': lambda x: -(p_obs * np.log(p_obs / x) + (1 - p_obs) * np.log((1 - p_obs) / (1 - x)) - Hx)},
            {'type': 'ineq',
             'fun': lambda x: (p_obs * np.log(p_obs / x) + (1 - p_obs) * np.log((1 - p_obs) / (1 - x)))}
            )
    bnds = [(lboudns, ubounds)]
    res = minimize(fun, x0=p_hat, jac=fun_deriv, constraints=cons, method='SLSQP',
                   bounds=bnds)
    return res.x[0]

# interval bounds
x_care = 1
p = np.mean(Intv_L[Intv_L['X']==x_care]['Y'])
p_hat = np.mean(Intv_S[Intv_S['X']==x_care]['Y'])
p_hat_sd = np.sqrt(p_hat * (1-p_hat))
t = stats.t.ppf(1-(1-0.99)/2,Ns)

lb_conf = p_hat - t*p_hat_sd/np.sqrt(Ns)
ub_conf = p_hat + t*p_hat_sd/np.sqrt(Ns)

p_obs = np.mean(Obs[Obs['X']==x_care]['Y'])
px = np.mean(Obs['X'])
Hx = partial_entropy(px,x_care)

lb_opt = causal_bound(p_obs,(lb_conf+ub_conf)/2,max(lb_conf,0.01),min(ub_conf,0.99),Hx=Hx, minmax_mode='min')
ub_opt = causal_bound(p_obs,(lb_conf+ub_conf)/2,max(lb_conf,0.01),min(ub_conf,0.99),Hx=Hx, minmax_mode='max')

lb = max(lb_opt,lb_conf)
ub = min(ub_opt,ub_conf)




