import random

import numpy as np
import matplotlib.pyplot as plt

alpha=[0.0001,0.00001,0.0000001]
N=1000
x = np.linspace(0, 1, N)
z = 20*np.sin(2*np.pi * 3 * x) + 100*np.exp(x)
error = 10 * np.random.randn(N)
t = z + error

shuffledX=np.copy(x)
random.shuffle(shuffledX)
trainX=np.copy(shuffledX[0:int(len(shuffledX)/2)])
validationX=np.copy(shuffledX[int(len(shuffledX)/2):int(len(shuffledX)*3/2)])
testX=np.copy(shuffledX[int(len(shuffledX)*3/2):int(len(shuffledX))])

shuffledZ=20*np.sin(2*np.pi*3*shuffledX)+100*np.exp(shuffledX)
shuffledT=shuffledZ+error

F = np.zeros((N, 24))
for i in range(N):
    for j in range(8):
        F[i, j] = np.power(x[i], j)
for i in range(N):
    for j in range(8,12):
        F[i,j]= np.power(np.sin(x[i]),j%7)
for i in range(N):
    for j in range(12,16):
        F[i,j]= np.power(np.cos(x[i]),j%11)
for i in range(N):
    for j in range(16,20):
        F[i,j]= np.power(np.sqrt(x[i]),j%15)
for i in range(N):
    for j in range(20,24):
        F[i,j]= np.power(np.exp(x[i]),j%19)
print(F)


