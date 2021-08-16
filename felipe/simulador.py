#! /usr/bin/env python3
# Felipe Derewlany Gutierrez - Simulador Onda Acústica
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from scipy import signal
import time


def laplaciano2(matriz):
    M = matriz.shape[0]
    N = matriz.shape[1]
    lap = np.zeros((M+2, N+2))
    lap[1:-1, 1:-1] = -4 * matriz

    lap[0:-2, 1:-1] += matriz
    lap[2:, 1:-1] += matriz

    lap[1:-1, 2:] += matriz
    lap[1:-1, 0:-2] += matriz

    y = lap[1:-1, 1:-1]
    return y


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


def laplaciano6(matriz):
    #f_xx = (2*f[i-3]-27*f[i-2]+270*f[i-1]-490*f[i+0]+270*f[i+1]-27*f[i+2]+2*f[i+3])/(180*1.0*h**2)
    M = matriz.shape[0]
    N = matriz.shape[1]
    lap = np.zeros((M+6, N+6))
    lap[3:-3, 3:-3] = -490 * matriz

    lap[0:-6, 3:-3] += 2 * matriz
    lap[1:-5, 3:-3] += -27 * matriz
    lap[2:-4, 3:-3] += 270 * matriz
    lap[4:-2, 3:-3] += 270 * matriz
    lap[5:-1, 3:-3] += -27 * matriz
    lap[6:, 3:-3] += 2 * matriz

    lap[3:-3, 0:-6] += 2 * matriz
    lap[3:-3, 1:-5] += -27 * matriz
    lap[3:-3, 2:-4] += 270 * matriz
    lap[3:-3, 4:-2] += 270 * matriz
    lap[3:-3, 5:-1] += -27 * matriz
    lap[3:-3, 6:] += 2 * matriz

    y = lap[3:-3, 3:-3]
    return y/180


def gera_video_otimizacao(nome, nit):
    metadata = dict(title='Simulação Ultrassom', author='Daniel Rossato')
    writer = matplotlib.animation.FFMpegWriter(fps=30, metadata=metadata)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 4))
    eixos = [1000 * x for x in [0, Nx * h, Ny * h, 0]]
    im1 = ax1.imshow(u2[:, :, 0], extent=eixos, vmin=0, vmax=1)
    im2 = ax1.imshow(meio, extent=eixos, alpha=0.1)
    im3 = ax2.imshow(u4[:, :, 0], extent=eixos, vmin=0, vmax=1)
    im4 = ax2.imshow(meio, extent=eixos, alpha=0.1)

    #plt.colorbar()
    #plt.xlabel('x [mm]')
    #plt.ylabel('z [mm]')

    with writer.saving(fig, f"D:/AUSPEX/videos/{nome}.mp4", 400):
        for n in range(0, nit):
            # Exibir
            im1.set_data(u2[:, :, n])
            ax1.set_title(f"u[{(n * ts):15.9f} s]")

            im3.set_data(u4[:, :, n])
            ax2.set_title(f"u[{(n * ts):15.9f} s]")

            writer.grab_frame()


# Dados da simulação
ts = 10e-9  # discretização temporal (cada instante equivale a 10 ns)
h = 200e-6  # discretização espacial (cada pixel equivale a 200 um)
#cmax = 5800  # Velocidade máxima no meio (aço)
#cmax = 1500  # Velocidade máxima no meio (água do mar)

# C = cmax*ts/h  # Condição CFL

Nx = 100  # quantidade de pixels x
Ny = 100  # quantidade de pixels y
Nt = 500  # quantidade de instantes de tempo

# Matriz que representa o meio
meio = np.zeros((Nx, Ny))
meio[0:36, 0:100] += 2500*np.ones((36, 100))  # Água do mar
meio[36:100, 0:100] += 5800*np.ones((64, 100))  # Aço

# Posição transdutor
x_trans = 50
y_trans = 25

# Inicialização dos vetores
u2 = np.zeros((Nx, Ny, Nt))  # u usando Laplaciano accuracy 2
u4 = np.zeros((Nx, Ny, Nt))  # u usando Laplaciano accuracy 4
u6 = np.zeros((Nx, Ny, Nt))  # u usando Laplaciano accuracy 6
V = np.zeros((Nx, Ny, Nt))  # Vij = ut ij 0 (derivada da condição inicial)

# Condições iniciais
V[:, :, :] = 0

# Termo de fonte
t = np.arange(0, (Nt-1)*ts, ts)
#f = np.zeros((1, Nt))
#f[0,0:50] = 1
f = signal.gausspulse(t, 5e6, 0.6)
#plt.ioff()
#plt.figure()
#plt.plot(t, f)
#plt.show()
termo_fonte = np.zeros((Nx, Ny, Nt))
termo_fonte[y_trans, x_trans, :] = f


# Calcula CFL para grid inteira
C2 = (meio[:, :] * ts / h) ** 2

start_time = time.time()

# Com Laplaciano accuracy 2
# Primeiro passo
u2[:, :, 1] = u2[:, :, 0] + ts * V[:, :, 0] + 0.5 * C2 * laplaciano2(u2[:, :, 0])\
    + 0.5 * ts ** 2 * termo_fonte[:, :, 0]

# Encontrar cada ponto no array de posições para cada tempo
for n in range(1, Nt-1):
    # Encontrar cada ponto no array de posições t[n+1]
    u2[:, :, n + 1] = 2 * u2[:, :, n] - u2[:, :, n - 1] + C2 * laplaciano2(u2[:, :, n]) \
        + termo_fonte[:, :, n]

    # Garantir condições nas bordas
    u2[0, :, n + 1] = 0
    u2[Nx - 1, :, n + 1] = 0
    u2[:, 0, n + 1] = 0
    u2[:, Ny - 1, n + 1] = 0

simulation_time1 = time.time()
tsim2 = simulation_time1 - start_time
print(f"Tempo para fazer a simulacao Laplaciano accuracy 2: {tsim2}")

# Com Laplaciano accuracy 4
# Primeiro passo
u4[:, :, 1] = u4[:, :, 0] + ts * V[:, :, 0] + 0.5 * C2 * laplaciano4(u4[:, :, 0])\
    + 0.5 * ts ** 2 * termo_fonte[:, :, 0]

# Encontrar cada ponto no array de posições para cada tempo
for n in range(1, Nt-1):
    # Encontrar cada ponto no array de posições t[n+1]
    u4[:, :, n + 1] = 2 * u4[:, :, n] - u4[:, :, n - 1] + C2 * laplaciano4(u4[:, :, n]) \
       + termo_fonte[:, :, n]

    # Garantir condições nas bordas
    u4[0, :, n + 1] = 0
    u4[Nx - 1, :, n + 1] = 0
    u4[:, 0, n + 1] = 0
    u4[:, Ny - 1, n + 1] = 0

simulation_time2 = time.time()
tsim4 = simulation_time2 - simulation_time1
print(f"Tempo para fazer a simulacao Laplaciano accuracy 4: {tsim4}")

# Com Laplaciano accuracy 6
# Primeiro passo
u6[:, :, 1] = u6[:, :, 0] + ts * V[:, :, 0] + 0.5 * C2 * laplaciano6(u6[:, :, 0])\
    + 0.5 * ts ** 2 * termo_fonte[:, :, 0]

# Encontrar cada ponto no array de posições para cada tempo
for n in range(1, Nt-1):
    # Encontrar cada ponto no array de posições t[n+1]
    u6[:, :, n + 1] = 2 * u6[:, :, n] - u6[:, :, n - 1] + C2 * laplaciano6(u6[:, :, n]) \
       + termo_fonte[:, :, n]

    # Garantir condições nas bordas
    u6[0, :, n + 1] = 0
    u6[Nx - 1, :, n + 1] = 0
    u6[:, 0, n + 1] = 0
    u6[:, Ny - 1, n + 1] = 0

simulation_time3 = time.time()
tsim6 = simulation_time3 - simulation_time2
print(f"Tempo para fazer a simulacao Laplaciano accuracy 6: {tsim6}")

# nomes arquivos
video_name = 'Pulso Gaussiano - Laplaciano 2 e 4'

# Gera um vídeo a partir de cada plot
gera_video_otimizacao(video_name, Nt)

video_time = time.time()
tvid = video_time - simulation_time3
print(f"Tempo para fazer o video: {tvid}")

end_time = time.time()
total_time = end_time - start_time
print(f"Tempo total: {total_time}")
