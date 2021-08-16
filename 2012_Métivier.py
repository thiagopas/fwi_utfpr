#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 10:27:12 2021

@author: danielpipa
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.ndimage.filters import correlate as imfilter

Lx = 2000
Ly = Lx
dx = 20
dy = dx
Nx = round(Lx/dx)
Ny = round(Ly/dy)
ny, nx = np.meshgrid(np.arange(Ny), np.arange(Nx))

c0 = 1500
c1 = 4500

c = c0*np.ones((Ny, Nx))
c[(nx-46)**2+(ny-46)**2 <= 25] = c1
c[(nx-54)**2+(ny-54)**2 <= 25] = c1

# Laplacian
deriv_order = 2
deriv_accuracy = 8
deriv_n_coef = 2*np.floor((deriv_order+1)/2).astype('int')-1+deriv_accuracy
p = np.round((deriv_n_coef-1)/2).astype('int')
A = np.arange(-p,p+1)**np.arange(0,2*p+1)[None].T
b = np.zeros(2*p+1)
b[deriv_order] = math.factorial(deriv_order)
h = np.zeros((deriv_n_coef,deriv_n_coef))
h[deriv_n_coef//2, :] = np.linalg.solve(A, b)
h += h.T

def lap(u):
    return imfilter(u, h)


# plt.imshow(c, origin='upper', extent=(0,Lx,Ly,0))
# plt.gca().xaxis.tick_top()

