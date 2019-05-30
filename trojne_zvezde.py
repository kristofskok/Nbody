#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:54:07 2019

@author: kristof
"""

import matplotlib.pyplot as plt
import numpy as np
P=75.316
a=17.83
e=0.967

G = 6.67408e-11
M = 1.989e30
r0 = 5 * 149597870700
vI = np.sqrt(G*M/r0)


def rKN(x, fx, n, hs):
    k1 = []
    k2 = []
    k3 = []
    k4 = []
    xk = []
    for i in range(n):
        k1.append(fx[i](x)*hs)
    for i in range(n):
        xk.append(x[i] + k1[i]*0.5)
    for i in range(n):
        k2.append(fx[i](xk)*hs)
    for i in range(n):
        xk[i] = x[i] + k2[i]*0.5
    for i in range(n):
        k3.append(fx[i](xk)*hs)
    for i in range(n):
        xk[i] = x[i] + k3[i]
    for i in range(n):
        k4.append(fx[i](xk)*hs)
    for i in range(n):
        x[i] = x[i] + (k1[i] + 2*(k2[i] + k3[i]) + k4[i])/6
    return x
#%%
def brezveze(x):
    return 1
def dvx(x, y):
    return 4*np.pi*x[1]*((y[1]-x[1])**2+(y[2]-x[2])**2+(y[3]-x[3])**2)**(-1.5)
def dvy(x, y):
    return 4*np.pi*x[2]*((y[1]-x[1])**2+(y[2]-x[2])**2+(y[3]-x[3])**2)**(-1.5)
def dvz(x, y):
    return 4*np.pi*x[3]*((y[1]-x[1])**2+(y[2]-x[2])**2+(y[3]-x[3])**2)**(-1.5)
def dRx(x):
    return 4*np.pi*x[4]
def dRy(x):
    return 4*np.pi*x[5]
def dRz(x):
    return 4*np.pi*x[6]

f=[brezveze, dRx, dRy, dRz, dvx, dvy, dvz]
#%%
dw = 1e-4
w_end = 10
N = round(w_end / dw)
du = dw/(2*N)
w = np.linspace(start=0, stop=N*dw, num=N, endpoint=True)
u = np.linspace(start=0, stop=N*du, num=N, endpoint=True)

arr = np.zeros((3, N, 6))
arr[0, 0] = [1, 0, 0, 0, 0.5, 0]
arr[1, 0] = [-1, 0, 0, 0, -0.5, 0]
arr[2, 0] = [0, 0, -100, 0, 0, 105000/vI]

pospesek = np.zeros((3,N,3))
vrtilna = np.zeros((3,N,3))

#najprej propagacija do w = dw/2
for i in range(0, N-1):
    vrtilna[:,i] = np.cross(arr[:,i,0:3], arr[:,i,3:6])
    arr[:,i+1,0:3] = arr[:,i,0:3] + du*4*np.pi * arr[:,i,3:6]

    for j in range(3):
        for k in range(3):
            if k!=j:
                pospesek[j,i] += 4*np.pi*np.linalg.norm(arr[j,i,0:3]-arr[k,i,0:3])**(-3) * (arr[k,i,0:3] - arr[j,i,0:3])

    arr[:,i+1,3:6] = arr[:,i,3:6] + du*pospesek[:,i]

#hitrosti nastavimo ob dw/2, oz. korak i=1/2
hitrosti = arr[:,-1,3:6].copy()
arr = np.zeros((3, N, 6))
arr[0, 0] = np.concatenate(([1, 0, 0], hitrosti[0]))
arr[1, 0] = np.concatenate(([-1, 0, 0], hitrosti[1]))
arr[2, 0] = np.concatenate(([0, 0, -100], hitrosti[2]))

pospesek = np.zeros((3,N,3))
vrtilna = np.zeros((3,N,3))

for i in range(0, N-1):
    vrtilna[:,i] = np.cross(arr[:,i,0:3], arr[:,i,3:6])
    arr[:,i+1,0:3] = arr[:,i,0:3] + du*4*np.pi * arr[:,i,3:6]

    for j in range(3):
        for k in range(3):
            if k!=j:
                pospesek[j,i] += 4*np.pi*np.linalg.norm(arr[j,i,0:3]-arr[k,i,0:3])**(-3) * (arr[k,i,0:3] - arr[j,i,0:3])

    arr[:,i+1,3:6] = arr[:,i,3:6] + du*pospesek[:,i]


#%%
telo = 0
koordinata = 1
fig, ax = plt.subplots(nrows=4, sharex=True)
ax[0].plot(w, pospesek[telo,:,koordinata])
ax[1].plot(w, arr[telo,:,koordinata + 3])
ax[2].plot(w, arr[telo,:,koordinata])
ax[3].plot(w, vrtilna[telo,:,koordinata]/vrtilna[telo,0,koordinata])
ax[0].set_title('telo='+str(telo)+' koordinata='+str(koordinata)+ ' dw='+str(dw)+' N='+str(N))
plt.show()
