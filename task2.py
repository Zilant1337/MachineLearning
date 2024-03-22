# train 60, validation 20, test 20
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import math


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


N = 1000
x = np.linspace(0, 1, N)
z = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
error = 10 * np.random.randn(N)
t = z + error

# NB. error тоже бы пошафлить

# Параметры тренировки
alphæ = [0.0001, 0.00001, 0.0000001]
PETER_COUNT = 20  # Кол-во планов для тренировки
FUN_NUM = 15

x_idxs = list(range(N))
random.shuffle(x_idxs)
x_shuffled = x[x_idxs]

tenth = N // 10
indices_train = np.arange(0, 6 * tenth)
indices_valid = np.arange(6 * tenth, 8 * tenth)
indices_test = np.arange(8 * tenth, 10 * tenth)

z_shuffled = 20 * np.sin(2 * np.pi * 3 * x_shuffled) + 100 * np.exp(x_shuffled)
t_shuffled = z_shuffled + error

F = np.zeros((N, 24))  # мастерплан

for i in range(N):
    for j in range(8):
        F[i, j] = np.power(x_shuffled[i], j + 1)
for i in range(N):
    for j in range(8, 12):
        F[i, j] = np.power(np.sin(x_shuffled[i]), j % 7)
for i in range(N):
    for j in range(12, 16):
        F[i, j] = np.power(np.cos(x_shuffled[i]), j % 11)
for i in range(N):
    for j in range(16, 20):
        F[i, j] = np.power(np.sqrt(x_shuffled[i]), j % 15)
for i in range(N):
    for j in range(20, 24):
        F[i, j] = np.power(np.exp(x_shuffled[i]), j % 19)

Ф = np.zeros((N, 24))  # мастерплан

for i in range(N):
    for j in range(8):
        Ф[i, j] = np.power(x[i], j + 1)
for i in range(N):
    for j in range(8, 12):
        Ф[i, j] = np.power(np.sin(x[i]), j % 7)
for i in range(N):
    for j in range(12, 16):
        Ф[i, j] = np.power(np.cos(x[i]), j % 11)
for i in range(N):
    for j in range(16, 20):
        Ф[i, j] = np.power(np.sqrt(x[i]), j % 15)
for i in range(N):
    for j in range(20, 24):
        Ф[i, j] = np.power(np.exp(x[i]), j % 19)

# Генерация планов по мастерплану — выбираем случайные столбики
idxs = []  # idxs[i] — индексы столбиков i-го плана в мастерплане: []
plans_full = []  # plans_full[i] — значения планов: np.array((,))
планс_фулл = []
while len(plans_full) != PETER_COUNT:
    plan_idxs = sorted(random.sample(range(24), k=FUN_NUM))
    if plan_idxs in idxs:
        continue
    idxs.append(plan_idxs)
    plans_full.append(F.T[plan_idxs].T)
    планс_фулл.append(Ф.T[plan_idxs].T)


def errorist(plan_no, plan_idxs, indices, w, alpha):
    plan_idxs = np.array(plan_idxs)
    subqsman = plans_full[plan_no][indices]
    print('w.shape =', w.shape)
    qsman1 = (t_shuffled[indices].T - subqsman @ w) ** 2
    print('qsman1.shape =', qsman1.shape)
    qsman2 = alpha * np.sum(np.abs(w) ** 2)
    print('qsman2.shape =', qsman2.shape)
    error = (sum(qsman1) + qsman2) / N
    print('error =', error)
    return error


min_error = math.inf
best_w = None
best_idxs = None
best_alpha = None
best_plan_no = None
for i in range(PETER_COUNT):
    plan = plans_full[i]
    idx = idxs[i]
    for alpha in alphæ:
        print(bcolors.OKGREEN + bcolors.BOLD + "ITER: alpha = " + str(alpha) + ", peter = " + str(i) + bcolors.ENDC)
        size = plan.shape[0]
        print('plan.shape =', plan.shape)
        print('t_shuffled[:size].shape = ', t_shuffled[:size].shape)
        inv = np.linalg.pinv
        w = (inv(plan @ plan.T + alpha * np.identity(size)) @ plan).T @ t_shuffled[:size]
        print(np.array(w).shape)

        error = errorist(i, idx, indices_valid, w, alpha)
        if error < min_error:
            best_plan_no = i
            best_w = w
            best_idxs = idx
            best_alpha = alpha
            min_error = error

print(bcolors.OKCYAN + "TRAINING COMPLETE" + bcolors.ENDC)

final_error = errorist(best_plan_no, best_idxs, indices_test, best_w, best_alpha)

print("шоколадная паста")

unshuffled = plans_full[best_plan_no] @ best_w
desorted = [unshuffled[x_idxs[i]] for i in range(N)]
паша = планс_фулл[best_plan_no] @ best_w

print(f"Best Alpha: {best_alpha}\nBest W: {best_w}\nBest plan: {best_idxs}\nMin Error: {min_error}")

fig = plt.figure()
plt.plot(x, t, 'r.')
plt.plot(x, z, 'g-')
plt.plot(x, паша, 'c.')
# тут график регрессии на test?
plt.show()
