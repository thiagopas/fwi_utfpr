#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 12:05:18 2020

@author: danielpipa

Model-based Ultrasound Inversion

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize, Bounds

Lx = 50e-3 # Block width
Lz = 50e-3 # Block height
dx = 1e-3 # x-axis step
dz = 1e-3 # z-axis step
Nx = round(Lx/dx)
Nz = round(Lz/dz)

Lt = 15e-6 # Simulation time
fs = 66.66e6 # Sampling frequency
dt = 1/fs # Sampling time
Nt = round(Lt/dt)

ad = np.sqrt((dx*dz)/(dt**2)) # Adimensionality constant

# Sound speeds
cMat = {}
cMat['aluminium'] = 6320/ad
cMat['copper'] = 4900/ad
cMat['steel'] = 5490/ad

# Velocity field
p1 = Nz//3
p2 = Nx//3
c = cMat['aluminium'] *np.ones((Nz, Nx))
c[p1:-p1,p2:-p2] = cMat['copper']
c[p1:-p1,round(Nx/2.2):-p2] = cMat['steel']

f0 = 4e6 # Transducer central frequency
t0 = 1e-6 # Pulse time

# Grid coordinates
[xgrid, zgrid] = np.meshgrid(np.arange(Nx), np.arange(Nz))

# Source location
xzs = np.full((Nz, Nx), False)

Ns = 11 # Number of sources
R = Nz//3
for i in range(Ns):
    theta = 2*np.pi*i/Ns
    z = int(round(R*np.sin(theta)+Nz/2))
    x = int(round(R*np.cos(theta)+Nz/2))
    xzs[z,x] = True

# Source signal
t = np.arange(0, Lt, dt)
def signal(t0):
    s = f0**2*(t-t0)*np.exp(-f0**2*(t-t0)**2)
    return s/np.max(s)

s = np.zeros((Ns,Nt))
for i in range(Ns):
    s[i,:] = signal((i+1)*t0)

# Measurement location
xzm = (xgrid==0) | (xgrid==Nx-1) | (zgrid==0) | (zgrid==Nz-1)

# Measurement points
xzm = np.full((Nz, Nx), False)
# Nm = 10
# xzm[Nm:-Nm,Nm:-Nm] = True

Nm = 11 # Number of sources
R = Nz//3
for i in range(Ns):
    theta = 2*np.pi*(i+.5)/Ns
    z = int(round(R*np.sin(theta)+Nz/2))
    x = int(round(R*np.cos(theta)+Nz/2))
    xzm[z,x] = True

# Gradient mask
gmask = np.full((Nz,Nx),0)
gmask[p1:-p1,p2:-p2] = 1

courant = c.max()
if courant > 1:
    raise ValueError("Courant error")
else:
    print("Courant OK: ",courant)

# Laplacian
def lap(u):
    v = -4*u.copy()
    v[:-1,:,...] += u[1:,:,...]
    v[1:,:,...] += u[:-1,:,...]
    v[:,:-1,...] += u[:,1:,...]
    v[:,1:,...] += u[:,:-1,...]
    return v

def simulate(c, s, xzs, u_ini=np.zeros((Nz,Nx,2))):
    u = np.zeros((Nz,Nx,Nt)) # Pressure field
    u[:,:,0:2] = u_ini
    u[xzs, 0:2] += s[:,0:2]
    for k in range(2,Nt):
        u[:,:,k] = 2*u[:,:,k-1] - u[:,:,k-2] + (c**2)*lap(u[:,:,k-1])
        u[xzs, k] += s[:,k]
    return u


def J(ch):
    ch = ch.reshape((Nz, Nx))
    u = simulate(ch, s, xzs) # Forward field
    # Gradient by Adjoint State Method
    g = (u-u0)[xzm, :] # u0 is the Measured field
    uu = simulate(ch, g[:,::-1], xzm) # Adjoint field
    # uu = np.flip(uu, axis=2)
    uu = simulate(ch, g, xzm, uu[:,:,-1:-3:-1])
    grd = gmask*np.sum(uu*lap(u),axis=2) # Gradient
    return np.sum(((u-u0)[xzm,:])**2), grd.ravel()

def play(u0):
    vmin = u0.min()/5
    vmax = u0.max()/5
    for t in range(0, Nt, 15):
        plt.clf()
        plt.imshow(u0[:, :, t], vmin=vmin, vmax=vmax, origin='lower')
        plt.xlabel(t)
        plt.draw()
        plt.pause(0.000001)


# Add measurement noise
def add_noise(u, db):
    s = np.sqrt(np.mean(u**2)*10**(-db/10))
    return u + s*np.random.randn(*u.shape)

u0_clean = simulate(c, s, xzs)
u0 = add_noise(u0_clean, 1e10)
# play(u0_clean)

#%%

ch = cMat['aluminium']*np.ones((Nz, Nx))

bnds = Bounds(min(cMat.values()), max(cMat.values()))
opt={'disp':True, 'gtol': 1e-9, 'ftol': 1e-9, 'maxiter': 150, 'maxcor': 1000}
mthd = 'L-BFGS-B'
r = minimize(J, ch.flatten(), jac=True, method=mthd, bounds=bnds, options=opt)
ch = r.x.reshape((Nz, Nx))

# Plotting

plt.figure(1)
im = plt.imshow(c*ad, cmap='plasma', origin='lower')
plt.colorbar()
plt.title('True Ultrasound Speed')
colors = [im.cmap(im.norm(value*ad)) for value in list(cMat.values())]
patches = [mpatches.Patch(color=colors[i], label=list(cMat.keys())[i]) for i in range(len(cMat.values()))]
plt.legend(handles=patches, loc=4, borderaxespad=0. )

plt.figure(2)
plt.imshow(ch*ad, cmap='plasma', origin='lower')
plt.colorbar()
plt.title('Estimated Ultrasound Speed')
plt.legend(handles=patches, loc=4, borderaxespad=0. )
