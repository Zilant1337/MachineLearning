
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import math

def E(planNumber, indices, w, alpha):
    subqsman = shuffledPlansFull[planNumber][indices]
    # print('w.shape =', w.shape)
    qsman1 = (shuffledT[indices].T - subqsman @ w) ** 2
    # print('qsman1.shape =', qsman1.shape)
    qsman2 = alpha * np.sum(np.abs(w) ** 2)
    # print('qsman2.shape =', qsman2.shape)
    error = (sum(qsman1) + qsman2) / N
    # print('error =', error)
    return error

N = 1000
x = np.linspace(0, 1, N)
z = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
error = 10 * np.random.randn(N)
t = z + error


lambdaValues = [0.0001, 0.00001, 0.0000001, 0.000000001, 0.00000000001]
iterationCount = 25  # Кол-во планов для тренировки
funcAmount = 15

xIds = list(range(N))
random.shuffle(xIds)
shuffledX = x[xIds]

tenth = N // 10
trainIds = np.arange(0, 6 * tenth)
validationIds = np.arange(6 * tenth, 8 * tenth)
testIds = np.arange(8 * tenth, 10 * tenth)

shuffledZ = 20 * np.sin(2 * np.pi * 3 * shuffledX) + 100 * np.exp(shuffledX)
shuffledT = shuffledZ + error

shuffledF = np.zeros((N, 24))  # мастерплан

for i in range(N):
    for j in range(8):
        shuffledF[i, j] = np.power(shuffledX[i], j+1)
for i in range(N):
    for j in range(8, 12):
        shuffledF[i, j] = np.power(np.sin(shuffledX[i]), j % 7)
for i in range(N):
    for j in range(12, 16):
        shuffledF[i, j] = np.power(np.cos(shuffledX[i]), j % 11)
for i in range(N):
    for j in range(16, 20):
        shuffledF[i, j] = np.power(np.sqrt(shuffledX[i]), j % 15)
for i in range(N):
    for j in range(20, 24):
        shuffledF[i, j] = np.power(np.exp(shuffledX[i]), j % 19)

F = np.zeros((N, 24))

for i in range(N):
    for j in range(8):
        F[i, j] = np.power(x[i], j + 1)
for i in range(N):
    for j in range(8, 12):
        F[i, j] = np.power(np.sin(x[i]), j % 7)
for i in range(N):
    for j in range(12, 16):
        F[i, j] = np.power(np.cos(x[i]), j % 11)
for i in range(N):
    for j in range(16, 20):
        F[i, j] = np.power(np.sqrt(x[i]), j % 15)
for i in range(N):
    for j in range(20, 24):
        F[i, j] = np.power(np.exp(x[i]), j % 19)

ids = []  # ids[i] — индексы столбиков i-го плана в мастерплане: []
shuffledPlansFull = []  # shuffledPlansFull[i] — значения планов: np.array((,))
plansFull = [] # plansFull[i] - значения планов по изначальным X
while len(shuffledPlansFull) != iterationCount:
    planIds = sorted(random.sample(range(24), k=funcAmount))
    if planIds in ids:
        continue
    ids.append(planIds)
    shuffledPlansFull.append(shuffledF.T[planIds].T)
    plansFull.append(F.T[planIds].T)




minError = math.inf
bestW = None
bestIds = None
bestAlpha = None
bestPlanNumber = None
for i in range(iterationCount):
    plan = shuffledPlansFull[i]
    idx = ids[i]
    for alpha in lambdaValues:
        size = plan.shape[0]
        inv = np.linalg.pinv
        w = (inv(plan @ plan.T + alpha * np.identity(size)) @ plan).T @ shuffledT[:size]

        error = E(i, validationIds, w, alpha)
        if error < minError:
            bestPlanNumber = i
            bestW = w
            bestIds = idx
            bestAlpha = alpha
            minError = error

finalError = E(bestPlanNumber, testIds, bestW, bestAlpha)

regression = plansFull[bestPlanNumber] @ bestW

print(f"Best Alpha: {bestAlpha}\nBest W: {bestW}\nBest plan: {bestIds}\nMin Error: {minError}")

fig = plt.figure()
plt.plot(x, t, 'b.')
plt.plot(x, z, 'y-')
plt.plot(x, regression, 'r')
plt.show()
