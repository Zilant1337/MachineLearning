import numpy as np
import matplotlib.pyplot as plt


def DoTask(M):
    global N,t,z,x
    F = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            F[i, j] = np.power(x[i], j)
    w = np.linalg.inv(np.transpose(F) @ F) @ np.transpose(F) @ t
    # w=np.linalg.pinv(F)@t
    # y=[]
    # for i in range(len(x)):
    #     yval=0
    #     for j in range(M):
    #         yval+=w[j]*np.power(x[i],j)
    #     y.append(yval)
    y = F @ w
    E = np.sum(np.power(t - y, 2)) / 2
    if(M==1 or M==8 or M==100):
        plt.plot(x,z,color="green")
        plt.plot(x,y,color="red")
        plt.scatter(x,t, color="blue")
        plt.show()
    return E

N=1000
x = np.linspace(0, 1, N)
z = 20*np.sin(2*np.pi * 3 * x) + 100*np.exp(x)
error = 10 * np.random.randn(N)
t = z + error
M=[1,8,100]
err=[]
for i in range(1,101):
    err.append(DoTask(i))
plt.plot(err)
plt.show()
