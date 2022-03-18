import numpy as np
from scipy.sparse.linalg import cg
from scipy.optimize import minimize


def simulate(xc, zc, r, c, x, z):
    dists_to_surface = np.sqrt((x - xc)**2 +
                               (z - zc)**2) - r
    t = dists_to_surface * 2 / c
    return t


p_true = np.array([22., 90., 70.])  # [xc, zc, r]
pitch = .6
c = 1486.

x = np.arange(64, dtype='float')
x = x - x.mean()
x = x * pitch
z = np.zeros_like(x)
t = simulate(p_true[0], p_true[1], p_true[2],
             c, x, z)

def g(p):
    u = simulate(p[0], p[1], p[2], c, x, z)
    return np.sum((u - t) ** 2)

n = t.shape[0]
m = 3  # [xc, zc, r]

p = np.array([35., 38., 2500.])  # initial guess
xc, zc, r = p
u = simulate(p[0], p[1], p[2], c, x, z)

df_du = np.eye(n) * (- c / 2)
df_dxc = (xc - x) / np.sqrt((x - xc)**2 + (z - zc)**2)
df_dzc = (zc - z) / np.sqrt((x - xc)**2 + (z - zc)**2)
df_dr = - np.ones(n, dtype=float)
df_dp = np.array([df_dxc, df_dzc, df_dr]).transpose()
dg_du = 2 * u - 2 * t

adj = cg(df_du.transpose(), dg_du)[0]
dg_dp = - np.matmul(adj, df_dp)
print(dg_dp)

def grad(p):
    u = simulate(p[0], p[1], p[2], c, x, z)

    xc, zc, r = p
    # df_du = np.eye(n) * (- c / 2) # No longer needed
    df_dxc = (xc - x) / np.sqrt((x - xc) ** 2 + (z - zc) ** 2)
    df_dzc = (zc - z) / np.sqrt((x - xc) ** 2 + (z - zc) ** 2)
    df_dr = - np.ones(n, dtype=float)
    df_dp = np.array([df_dxc, df_dzc, df_dr]).transpose()
    dg_du = 2 * u - 2 * t

    # inv_dfdu = np.eye(n, dtype=float) * (- 2 / c)
    inv_dfdu = - 2. / c

    #adj = np.matmul(dg_du, inv_dfdu)
    adj = dg_du * inv_dfdu

    dg_dp = - np.matmul(adj, df_dp)
    return dg_dp

options = {'maxiter' : 300}
sol = minimize(fun=g, x0=p, method='BFGS', tol=1e-30, jac=grad, options=options)
print('Solution: ' + str(sol.x))

# Initial guess