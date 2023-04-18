import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn import datasets
import numpy as np

mean1 = [-3, 3] # 均值
cov = [[0.4, 0], [0, 0.4]] # 协方差矩阵
s = 6
# 生成二维高斯随机数
X_train1 = np.random.multivariate_normal(mean1, cov, s)
mean2 = [3, -3] # 均值
# 生成二维高斯随机数
X_train2 = np.random.multivariate_normal(mean2, cov, s)
r = np.array([[-1,0.5],[-1,-2.5],[1,2.5],[1,-0.5]])
X_train = np.insert(X_train2, 0, X_train1, axis=0)
X_train = np.insert(X_train, 0, r, axis=0)
print(X_train)
v = np.array([1,-1,1,-1])
Y = -1*np.ones(s)
Y =  np.insert(Y, 0, np.ones(s), axis=0)
Y =  np.insert(Y, 0, v, axis=0)
print(Y)
Y_train = np.reshape(Y, (len(Y), 1))

print(X_train)

idx0 = np.where(Y_train == -1)
idx1 = np.where(Y_train == 1)
print(idx0)
print(X_train[idx0, 0])
plt.plot(X_train[idx0[0], 0], X_train[idx0[0], 1], 'rx')
plt.plot(X_train[idx1[0], 0], X_train[idx1[0], 1], 'bo')
plt.show()