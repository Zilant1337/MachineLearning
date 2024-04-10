from sklearn.datasets import fetch_california_housing
from numba import jit,cuda
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import math

@jit(target_backend='cuda')
def gradient_calculation(w):
    global t, alphaValue, x
    F=(x.T[trainIds]).T
    return -(t.T@F).T+(w.T@(F.T@F)).T+alphaValue*w.T



alphaValue =0.00000000001
x_src=fetch_california_housing().data
z=fetch_california_housing().target
N=len(x_src[:,0])

tenth = N // 10
trainIds = np.arange(0, 7 * tenth)
testIds = np.arange(7 * tenth, 10 * tenth)

# Нормализация
stddevs = np.std(x_src, 0)

means = np.mean(x_src,0)

x = (x_src - means)/stddevs

iteration_count = 25  # Кол-во планов для тренировки
learning_rate = 0.01


stddev=0.1
w_list=[]
w=[]
for i in range(len(x[0])):
    w.append(np.random.normal(0, stddev))
w_list.append(w)

for i in np.arange(1,1+learning_rate*(iteration_count),learning_rate):
    w_list.append(w_list[-1]-i*gradient_calculation(w_list[-1]))


