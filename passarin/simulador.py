#! /usr/bin/env python3
# Felipe Derewlany Gutierrez - Simulador Onda Acústica
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from scipy import signal
import time

def laplaciano4(matriz):
    #f_xx = (-1*f[i-2]+16*f[i-1]-30*f[i+0]+16*f[i+1]-1*f[i+2])/(12*1.0*h**2)
    M = matriz.shape[0]
    N = matriz.shape[1]
    lap = np.zeros((M+4, N+4))
    lap[2:-2, 2:-2] = -60 * matriz

    lap[0:-4, 2:-2] += -1 * matriz
    lap[1:-3, 2:-2] += 16 * matriz
    lap[3:-1, 2:-2] += 16 * matriz
    lap[4:, 2:-2] += -1 * matriz

    lap[2:-2, 0:-4] += -1 * matriz
    lap[2:-2, 1:-3] += 16 * matriz
    lap[2:-2, 3:-1] += 16 * matriz
    lap[2:-2, 4:] += -1 * matriz

    y = lap[2:-2, 2:-2]
    return y/12

def lap4(matriz):
    #f_xx = (-1*f[i-2]+16*f[i-1]-30*f[i+0]+16*f[i+1]-1*f[i+2])/(12*1.0*h**2)
    M = matriz.shape[0]
    N = matriz.shape[1]
    if matriz.shape.__len__() < 3:
        T = 1
    else:
        T = matriz.shape[2]
    lap = np.zeros((M + 4, N + 4, T))
    lap[2:-2, 2:-2, :] = -60 * matriz.reshape((M, N, T))

    lap[0:-4, 2:-2, :] += -1 * matriz.reshape((M, N, T))
    lap[1:-3, 2:-2, :] += 16 * matriz.reshape((M, N, T))
    lap[3:-1, 2:-2, :] += 16 * matriz.reshape((M, N, T))
    lap[4:, 2:-2, :] += -1 * matriz.reshape((M, N, T))

    lap[2:-2, 0:-4, :] += -1 * matriz.reshape((M, N, T))
    lap[2:-2, 1:-3, :] += 16 * matriz.reshape((M, N, T))
    lap[2:-2, 3:-1, :] += 16 * matriz.reshape((M, N, T))
    lap[2:-2, 4:, :] += -1 * matriz.reshape((M, N, T))

    y = lap[2:-2, 2:-2, :]
    return y/12


def simulacao(meio, termo_fonte=None, h=200e-6, ts=10e-9):
    # Dados da simulação

    # C = cmax*ts/h  # Condição CFL

    Nx = meio.shape[0]
    Ny = meio.shape[1]
    Nt = 1000  # quantidade de instantes de tempo

    # Posição transdutor
    x_trans = 50
    y_trans = 25

    # Inicialização dos vetores
    u4 = np.zeros((Nx, Ny, Nt))  # u usando Laplaciano accuracy 4
    V = np.zeros((Nx, Ny, Nt))  # Vij = ut ij 0 (derivada da condição inicial)

    # Condições iniciais
    V[:, :, :] = 0

    if termo_fonte is None:
        # Termo de fonte
        t = np.arange(0, Nt, dtype=float)
        t = t * ts
        f = signal.gausspulse(t-1.5e-6, 2.5e6, 0.5)
        termo_fonte = np.zeros((Nx, Ny, Nt))
        termo_fonte[y_trans, x_trans, :] = f


    # Calcula CFL para grid inteira
    C2 = (meio[:, :] * ts / h) ** 2

    start_time = time.time()

    # Com Laplaciano accuracy 4
    # Primeiro passo
    u4[:, :, 1] = u4[:, :, 0] + ts * V[:, :, 0] + 0.5 * C2 * laplaciano4(u4[:, :, 0])\
        + 0.5 * ts ** 2 * termo_fonte[:, :, 0]

    # Encontre cada ponto no array de posições para cada tempo
    for n in range(1, Nt-1):
        # Encontrar cada ponto no array de posições t[n+1]
        u4[:, :, n + 1] = 2 * u4[:, :, n] - \
                          u4[:, :, n - 1] + \
                          C2 * lap4(u4[:, :, n]).reshape((Nx, Ny)) + \
                          termo_fonte[:, :, n]

        # Garantir condições nas bordas
        u4[0, :, n + 1] = 0
        u4[Nx - 1, :, n + 1] = 0
        u4[:, 0, n + 1] = 0
        u4[:, Ny - 1, n + 1] = 0

    return u4

# Matriz que representa o meio
Nx = 100  # quantidade de pixels x
Ny = 100  # quantidade de pixels y
ts = 10e-9
h = 200e-6
meio = np.zeros((Nx, Ny))
meio[0:36, 0:100] += 4000*np.ones((36, 100))
meio[36:100, 0:100] += 5800*np.ones((64, 100))
meio[60:80, 60:80] = 4000
meio_true = np.copy(meio)
u_true = simulacao(meio, h=h, ts=ts)

meio = np.ones((Nx, Ny))*5000.

maxit = 70
dist = np.zeros(maxit, dtype=float)
t_old = time.time()
for n in range(maxit):
    print('Iteration ' + str(n) + ' of ' + str(maxit))
    u_guess = simulacao(meio, h=h, ts=ts)
    u_res = u_guess - u_true
    u_adj = simulacao(meio, termo_fonte=u_res, h=h, ts=ts)
    grad = np.sum(u_adj * lap4(u_true)/(h**2), 2) * ts
    meio = meio - grad/100

    difference = meio - meio_true
    difference[25, 50] = 0
    dist[n] = np.linalg.norm(difference)

    t_now = time.time()
    t_dif = t_now - t_old
    print('Elapsed time: ' + str(t_dif) + '. Estimated time: ' + str((maxit-n)*t_dif))
    t_old = t_now
plt.figure()
plt.subplot(121)
plt.imshow(meio_true, vmin=2000, vmax=7000)
plt.colorbar()
plt.title('True model')
plt.subplot(122)
plt.imshow(meio, vmin=2000, vmax=7000)
plt.colorbar()
plt.title('Estimated model')

plt.figure()
plt.plot(dist, 'o-')
plt.xlabel('iteration')
plt.ylabel(['error'])
plt.grid()