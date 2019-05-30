#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:54:07 2019

@author: kristof
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
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
dw = 1e-2
w_end = 3
N = round(w_end / dw)
w = np.linspace(start=0, stop=N*dw, num=N, endpoint=True)

arr = np.zeros((3, N, 6))
arr[0, 0] = [1, 0, 0, 0, 0.5, 0]
arr[1, 0] = [-1, 0, 0, 0, -0.5, 0]
arr[2, 0] = [0, 0, -20, 0, 0, 0/vI]

pospesek = np.zeros((3,N,3))
vrtilna = np.zeros((3,N,3))

for i in range(0, N-1):
    vrtilna[:,i] = np.cross(arr[:,i,0:3], arr[:,i,3:6])
    arr[:,i+1,0:3] = arr[:,i,0:3] + dw*4*np.pi * arr[:,i,3:6]

    for j in range(3):
        for k in range(3):
            if k!=j:
                pospesek[j,i] += 4*np.pi*np.linalg.norm(arr[j,i,0:3]-arr[k,i,0:3])**(-3) * (arr[k,i,0:3] - arr[j,i,0:3])

    arr[:,i+1,3:6] = arr[:,i,3:6] + dw*pospesek[:,i]

vrtilna[:,-1] = np.cross(arr[:,-1,0:3], arr[:,-1,3:6])
#%%
telo = 2
koordinata = 2
fig, ax = plt.subplots(nrows=4, sharex=True)
ax[0].plot(w, pospesek[telo,:,koordinata])
ax[1].plot(w, arr[telo,:,koordinata + 3])
ax[2].plot(w, arr[telo,:,koordinata])
ax[3].plot(w, vrtilna[telo,:,koordinata]/np.linalg.norm(vrtilna[telo,0]))
ax[0].set_title('telo='+str(telo)+' koordinata='+str(koordinata)+ ' dw='+str(dw)+' N='+str(N))
plt.show()

#%%
telo = 1
koordinata = 2
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
ax.plot(arr[telo,:,0], arr[telo,:,1])
plt.show()
#%%
from mpl_toolkits.mplot3d import Axes3D
telo = 0
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')
ax.plot(zs = w, xs = arr[0,:,0], ys = arr[0,:,1], c='b')
ax.plot(zs = w, xs = arr[1,:,0], ys = arr[1,:,1], c='r')
plt.show()
#%%
meja = 3

#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=15, bitrate=1800)
def update(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[num,0:2])
        line.set_3d_properties(data[num, 2])
        print(num)
    return lines

fig = plt.figure()
ax = p3.Axes3D(fig)
#ax = fig.add_subplot(111, projection='3d')

lines = [ax.plot(arr[ii, 0:1, 0], arr[ii, 0:1, 1], arr[ii, 0:1, 2], marker='o',markersize=9)[0] for ii in range(3)]

ax.set_xlim3d([-meja, meja])
ax.set_xlabel('X')

ax.set_ylim3d([-meja, meja])
ax.set_ylabel('Y')

ax.set_zlim3d([0, 3])
ax.set_zlabel('Z')

#ax.set_aspect('equal')
#ax.set_title('3D Test')

#ln0, = ax.plot(xs=arr[0, 0:1, 0], ys=arr[0, 0:1, 1], zs=arr[0, 0:1, 2])



ani = animation.FuncAnimation(fig, update, frames=np.arange(0,N-1), fargs=(arr, lines), interval=50, blit=False)

#ani.save('anim3.mp4', writer=writer, dpi=70)
plt.show()