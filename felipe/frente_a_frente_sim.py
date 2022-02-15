#! /usr/bin/env python3
# Port do algoritmo do Prof Pipa

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage.filters import correlate
import time
import math
import pyqtgraph as pg
from pyqtgraph.widgets.RawImageWidget import RawImageGLWidget


def speed_measuring(z1, z2, pressure_at_z1, pressure_at_z2, dh, medium_speed, mode=0):

    '''
    # Wave Speed Estimation [pixel/iteration]
    delta_pixels1 = z2 - z1
    u_delta_iterations1 = np.where(pressure_at_z2 == max(pressure_at_z2))[0] - \
                          np.where(pressure_at_z1 == max(pressure_at_z1))[0]
    u_wave_speed1pi = delta_pixels1 / u_delta_iterations1

    # Wave Speed Estimation [m/s]
    space1 = (z2 - z1) * dh
    u_propagation_time1 = t[np.where(pressure_at_z2 == max(pressure_at_z2))[0]] - \
                          t[np.where(pressure_at_z1 == max(pressure_at_z1))[0]]
    u_wave_speed1 = space1 / u_propagation_time1
    '''

    #print(f"\nMedia {medium_speed}")
    #print(f"Lap {deriv_accuracy} - Measured Wave Speed: {u_wave_speed1pi} pixels/timestep")
    #print(f"Lap {deriv_accuracy} - Measured Wave Speed: {u_wave_speed1} m/s")

    if mode == 0:
        # Wave Speed Estimation [pixel/iteration]
        delta_pixels1 = z2 - z1

        envelope1 = np.abs(signal.hilbert(pressure_at_z1))
        envelope2 = np.abs(signal.hilbert(pressure_at_z2))

        u_delta_iterations1 = np.where(envelope2 == max(envelope2))[0] - \
                              np.where(envelope1 == max(envelope1))[0]
        u_wave_speed1pi = delta_pixels1 / u_delta_iterations1

        print(f"Lap {deriv_accuracy} - Measured Wave Speed: {u_wave_speed1pi} pixels/timestep")
        return u_wave_speed1pi
    elif mode == 1:
        # Wave Speed Estimation [m/s]
        space1 = (z2 - z1) * dh

        envelope1 = np.abs(signal.hilbert(pressure_at_z1))
        envelope2 = np.abs(signal.hilbert(pressure_at_z2))

        u_propagation_time1 = t[np.where(envelope2 == max(envelope2))[0]] - \
                              t[np.where(envelope1 == max(envelope1))[0]]
        u_wave_speed1 = space1 / u_propagation_time1

        print(f"Lap {deriv_accuracy} - Measured Wave Speed: {u_wave_speed1} m/s")
        return u_wave_speed1
    else:
        return -1


# Parameters
T = 0.00001  # [s]
Lx = 0.01  # [m]
Lz = 0.07  # [m]
dt = 5e-9  # [s/iteration]
dx = 10e-5  # [m/pixel]
dz = dx  # [m/pixel]
Nt = math.ceil(T / dt)
Nx = math.ceil(Lx / dx)
Nz = math.ceil(Lz / dz)

print(f"{Nx}x{Nz} Grid x {Nt} iterations - dx = {dx} m x dz = {dz} m x dt = {dt} s")

ad = math.sqrt((dx * dz) / (dt ** 2))  # Adimensionality constant

#soundspeed = 1000  # [m/s]
#soundspeed = 1481  # [m/s]
soundspeed = 2500  # [m/s]
#soundspeed = 3000  # [m/s]
#soundspeed = 5800  # [m/s]
#soundspeed = 6000  # [m/s]

c = soundspeed / ad

CFL = soundspeed * dt / dx
print(f"CFL condition = {CFL}")

# Pressure Fields
u = np.zeros((Nz, Nx))
u_1 = np.zeros((Nz, Nx))
u_2 = np.zeros((Nz, Nx))

u_at_transducer = np.zeros(Nt)
u_at_poi1 = np.zeros(Nt)
u_at_poi2 = np.zeros(Nt)
u_at_poi3 = np.zeros(Nt)
u_at_poi4 = np.zeros(Nt)
u_at_poi5 = np.zeros(Nt)
u_at_poi6 = np.zeros(Nt)
u_at_poi7 = np.zeros(Nt)
u_at_poi8 = np.zeros(Nt)
u_at_poi9 = np.zeros(Nt)
u_at_poi10 = np.zeros(Nt)

# Sources
t = np.linspace(0, T - dt, Nt)  # Time array
frequency = 2e6  # [Hz]
delay = 1e-6
bandwidth = 0.6

f = signal.gausspulse(t - delay, frequency, bandwidth)

# Laplacian Kernels Stencil Calculation - Prof. Pipa
deriv_order = 2
deriv_accuracy = 2
deriv_n_coef = 2 * np.floor((deriv_order + 1) / 2).astype('int') - 1 + deriv_accuracy
p = np.round((deriv_n_coef - 1) / 2).astype('int')
A = np.arange(-p, p + 1) ** np.arange(0, 2 * p + 1)[None].T
b = np.zeros(2 * p + 1)
b[deriv_order] = math.factorial(deriv_order)
coeff = np.zeros((deriv_n_coef, deriv_n_coef))
# Solve system A*w = b
coeff[deriv_n_coef // 2, :] = np.linalg.solve(A, b)
coeff += coeff.T
# print(str(np.size(coeff, 0)) + " x " + str((np.size(coeff, 1))))

# Simulation Parameters
delta = 3e-3
z_f = round(Nz / 2)  # Transductor y coordinate
x_f = round(Nx - 1)
x_pois = 0
z_poi1 = round(Nz * 0.1)
z_poi2 = round(Nz * 0.2)
z_poi3 = round(Nz * 0.3)
z_poi4 = round(Nz * 0.4)
z_poi5 = round(Nz * 0.5)
z_poi6 = round(Nz * 0.6)
z_poi7 = round(Nz * 0.7)
z_poi8 = round(Nz * 0.8)
z_poi9 = round(Nz * 0.9)
z_poi10 = round(Nz - 1)

measured_speed1_pi = []
measured_speed1_ms = []
measured_speed2_pi = []
measured_speed2_ms = []


# Exhibition setup
sfmt = pg.QtGui.QSurfaceFormat()
sfmt.setSwapInterval(0)
pg.QtGui.QSurfaceFormat.setDefaultFormat(sfmt)

app = pg.QtGui.QApplication([])
riw = pg.widgets.RawImageWidget.RawImageGLWidget()
riw.show()

start_time = time.time()

x = np.zeros((Nz, 2 * Nx))

for k in range(4, Nt):
    iteration_start = time.time()

    u_0 = u
    u_2 = u_1
    u_1 = u_0

    #lap = signal.fftconvolve(u_1[:, :], coeff, mode='same')
    lap = correlate(u_1[:, :], coeff)

    u = (2 * u_1[:, :] - (1 - delta / 2) * u_2[:, :] + (c ** 2) * lap) / (1 + delta / 2)

    u[z_f, x_f] += f[k]

    # Speed measuring
    u_at_transducer[k] = u[z_f, x_f]
    u_at_poi1[k] = u[z_poi1, x_pois]
    u_at_poi2[k] = u[z_poi2, x_pois]
    u_at_poi3[k] = u[z_poi3, x_pois]
    u_at_poi4[k] = u[z_poi4, x_pois]
    u_at_poi5[k] = u[z_poi5, x_pois]
    u_at_poi6[k] = u[z_poi6, x_pois]
    u_at_poi7[k] = u[z_poi7, x_pois]
    u_at_poi8[k] = u[z_poi8, x_pois]
    u_at_poi9[k] = u[z_poi9, x_pois]
    u_at_poi10[k] = u[z_poi10, x_pois]

    # Tracking
    math_time = time.time()
    print(f"{k} / {Nt} - Math Time: {math_time - iteration_start} s")

    # Exhibition - QT
    riw.setImage(u.T, levels=[-0.1, 0.1])
    app.processEvents()

app.exit()

end_time = time.time()
total_time = end_time - start_time


print(f"\n\n{Nx}x{Nz} Grid x {Nt} iterations - dx = {dx} m x dz = {dz} m x dt = {dt} s")

print("\nMeasured Soundspeed for media 1 [pixels/timestep]")
measured_speed1_pi.append(speed_measuring(z_f, z_poi1, u_at_transducer, u_at_poi1, dz, soundspeed, mode=0))
measured_speed1_pi.append(speed_measuring(z_f, z_poi2, u_at_transducer, u_at_poi2, dz, soundspeed, mode=0))
measured_speed1_pi.append(speed_measuring(z_f, z_poi3, u_at_transducer, u_at_poi3, dz, soundspeed, mode=0))

print("\nMeasured Soundspeed for media 1 [m/s]")
measured_speed1_ms.append(speed_measuring(z_f, z_poi1, u_at_transducer, u_at_poi1, dz, soundspeed, mode=1))
measured_speed1_ms.append(speed_measuring(z_f, z_poi2, u_at_transducer, u_at_poi2, dz, soundspeed, mode=1))
measured_speed1_ms.append(speed_measuring(z_f, z_poi3, u_at_transducer, u_at_poi3, dz, soundspeed, mode=1))

print(f"\nMedia {soundspeed}")
print(f"Lap {deriv_accuracy} - Measured Wave Speed: {np.mean(measured_speed1_pi)} pixels/timestep")
print(f"Lap {deriv_accuracy} - Measured Wave Speed: {np.mean(measured_speed1_ms)} m/s")

print(f"\n\nTotal Time: {total_time} s")
print(f"Total Time: {total_time / 60} min")

plt.ioff()
plt.figure(1)
plt.plot(t, f, 'b', label='Fonte')
plt.plot(t, u_at_transducer, 'r', label='Medido')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude do sinal')
plt.title("Pressão no pixel do transdutor")
plt.legend()
plt.show()

plt.figure(2)
plt.plot(t, u_at_transducer, 'r', label='Medido na fonte')
plt.plot(t, u_at_poi1, label='Medido em outro pixel')
plt.plot(t, u_at_poi2)
plt.plot(t, u_at_poi3)
plt.plot(t, u_at_poi4)
plt.plot(t, u_at_poi5)
plt.plot(t, u_at_poi6)
plt.plot(t, u_at_poi7)
plt.plot(t, u_at_poi8)
plt.plot(t, u_at_poi9)
plt.plot(t, u_at_poi10)
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude do sinal')
plt.title(f"Pressão no pixel do transdutor vs nos pontos de interesse - Lap {deriv_accuracy}")
plt.legend()
plt.show()


envelopef = np.abs(signal.hilbert(u_at_transducer))
plt.figure(3)
plt.plot(t, u_at_transducer, 'b', label='Medido na fonte')
plt.plot(t, envelopef, 'g', label='Envelope')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude do sinal')
plt.title(f"Pressão no pixel do transdutor")
plt.legend()
plt.show()

envelopef = np.abs(signal.hilbert(u_at_poi9))
plt.figure(4)
plt.plot(t, u_at_poi9, 'b', label='Medido no ponto 9')
plt.plot(t, envelopef, 'g', label='Envelope')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude do sinal')
plt.title(f"Pressão no pixel poi9")
plt.legend()
plt.show()

plt.figure(5)
plt.plot(t, u_at_transducer/np.linalg.norm(u_at_transducer), 'b', label='Medido na fonte')
plt.plot(t, np.abs(signal.hilbert(u_at_transducer))/np.linalg.norm(u_at_transducer), 'g', label='Envelope')
plt.plot(t, u_at_poi9/np.linalg.norm(u_at_poi9), 'r', label='Medido no ponto 9')
plt.plot(t, envelopef/np.linalg.norm(u_at_poi9), 'c', label='Envelope')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude do sinal')
plt.title(f"Medição da velocidade")
plt.legend()
plt.show()
