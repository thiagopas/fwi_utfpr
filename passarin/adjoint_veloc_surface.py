import numpy as np
from scipy.sparse.linalg import cg
from scipy.optimize import minimize


def simulate(a, b, c, x):
    theta = np.arctan(a)
    t = (2 / c) * (a * x + b) * np.cos(theta)
    return t


p_true = np.array([np.tan(np.deg2rad(38.)), 35., 2500.])  # [a, b, c]
pitch = .6

x = np.arange(64, dtype='float')
x = x - x.mean()
x = x * pitch
z = np.zeros_like(x)
t = simulate(p_true[0], p_true[1], p_true[2], x)

def g(p):
    u = simulate(p[0], p[1], p[2], x)
    return np.sum((u - t) ** 2)

n = t.shape[0]
m = 3  # [xc, zc, r]

p = np.array([np.tan(np.deg2rad(38.)), 34., 2400.])  # initial guess
xc, zc, r = p
u = simulate(p[0], p[1], p[2], x)


def grad(p):
    u = simulate(p[0], p[1], p[2], x)

    a, b, c = p
    atan_a = np.arctan(a)
    df_da = 2 * x * (np.cos(atan_a) - a * np.sin(atan_a) / (1 + (a**2))) / c + \
            (2 * b / c) * (- np.sin(atan_a) / (1 + (a**2)))
    df_db = np.ones_like(df_da, dtype=float) * (2 / c) * np.cos(atan_a)
    df_dc = (- 2 / (c**2)) * np.cos(atan_a) * (a * x + b)
    df_dp = np.array([df_da, df_db, df_dc]).transpose()
    dg_du = 2 * u - 2 * t

    # inv_dfdu = np.eye(n, dtype=float) * (- 2 / c)
    inv_dfdu = - 1.

    #adj = np.matmul(dg_du, inv_dfdu)
    adj = dg_du * inv_dfdu

    dg_dp = - np.matmul(adj, df_dp)
    return dg_dp

mygrad = grad(p)
print('My grad: ' + str(np.rad2deg(np.arctan(mygrad[0]))) + ', ' + str(mygrad[1]) + ', ' + str(mygrad[2]))

options = {'maxiter' : 300, 'disp' : True}
sol = minimize(fun=g, x0=p, method='L-BFGS-B', tol=1e-30, jac=grad, options=options)
print('Solution: ' + str(np.rad2deg(np.arctan(sol.x[0]))) + ', ' + str(sol.x[1]) + ', ' + str(sol.x[2]))

# Initial guess