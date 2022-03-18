import numpy as np
from scipy.sparse.linalg import cg
from scipy.optimize import minimize


def simulate(b, c, x):
    theta = np.arctan(a)
    t = (2 / c) * (a * x + b) * np.cos(theta)
    return t


### Generate synthetic times of flight t ##############
p_true = np.array([35., 2500.])  # [b, c]
pitch = .6
a = np.tan(np.deg2rad(40.))
x = np.arange(64, dtype='float')
x = x - x.mean()
x = x * pitch
t = simulate(p_true[0], p_true[1], x)
#######################################################

def g(p):
    u = simulate(p[0], p[1], x)
    return np.sum((u - t) ** 2)

n = t.shape[0]
m = 2  # [b, c]

p = np.array([25., 3000.])  # initial guess
u = simulate(p[0], p[1], x)


def grad(p):
    u = simulate(p[0], p[1], x)

    b, c = p
    atan_a = np.arctan(a)
    df_db = np.ones_like(x, dtype=float) * (2 / c) * np.cos(atan_a)
    df_dc = (- 2 / (c**2)) * np.cos(atan_a) * (a * x + b)
    df_dp = np.array([df_db, df_dc]).transpose()
    dg_du = 2 * u - 2 * t

    # inv_dfdu = np.eye(n, dtype=float) * (- 2 / c)
    inv_dfdu = - 1.

    #adj = np.matmul(dg_du, inv_dfdu)
    adj = dg_du * inv_dfdu

    dg_dp = - np.matmul(adj, df_dp)
    return dg_dp


options = {'maxiter': 300, 'disp': False}
sol = minimize(fun=g, x0=p, method='BFGS', tol=1e-30, jac=grad, options=options)
print('Solution: ' + str(sol.x[0]) + ', ' + str(sol.x[1]))

# Initial guess