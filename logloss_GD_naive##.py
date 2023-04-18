import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import cvxpy as cp
def dset():
    mean1 = [-3, 3]  # 均值
    cov = [[0.4, 0], [0, 0.4]]  # 协方差矩阵
    s = 6
    # 生成二维高斯随机数
    X_train1 = np.random.multivariate_normal(mean1, cov, s)
    mean2 = [3, -3]  # 均值
    # 生成二维高斯随机数
    X_train2 = np.random.multivariate_normal(mean2, cov, s)
    r = np.array([[-0.5, 1], [0.5, -1], [1.5, 3], [-1.5, -3]])
    X_train = np.insert(X_train2, 0, X_train1, axis=0)
    X_train = np.insert(X_train, 0, r, axis=0)
    print(X_train)
    v = np.array([1, -1, 1, -1])
    Y = -1 * np.ones(s)
    Y = np.insert(Y, 0, np.ones(s), axis=0)
    Y = np.insert(Y, 0, v, axis=0)
    print(Y)
    Y_train = np.reshape(Y, (len(Y), 1))
    return  X_train,Y_train


# Generate a toy dataset
n_samples = 16
dim = 2
X_train, Y_train = dset()

print(X_train)
n_samples=n_samples+4
n_features_list = np.arange(3, dim+1, 10)
costs =[]
distances2 = []

def SVM(x_train, y_train):
    # SVM

    s_s = cp.Variable((x_train.shape[1], 1))
    # b_s = cp.Variable()

    objective = cp.Minimize(0.5*cp.norm(s_s) ** 2)
    constraints = [cp.multiply(y_train, x_train @ s_s) >= 1]
    prob = cp.Problem(objective, constraints)

    prob.solve()
    s_s_value = s_s.value
    # b_s_value = b_s.value
    return s_s_value

def LogisticLoss(w, X, y, lam):
    # Computes the cost function for all the training samples
    m = X.shape[0]
    Xw = np.matmul(X,w)
    yT = y.reshape(-1,1)
    yXw = np.multiply(yT,Xw)

    f = np.mean(np.logaddexp(0, -yXw))
    gMul = np.exp(-yXw)/(1 + np.exp(-yXw))
    ymul = -1*yT*gMul
    g = np.matmul(ymul.reshape(1, -1), X)
    g = g.reshape(-1, 1)
    return [f, g]

def model_logloss(x_train, y_train, num_iterations=20000, learning_rate=0.5):
    w_log = np.zeros((x_train.shape[1], 1))  # GRADED FUNCTION: initialize_with_zeros

    # Gradient descent
    # GRADED FUNCTION: optimize
    for i in range(num_iterations):
        # GRADED FUNCTION: propagate

        [cost, dw_log] = LogisticLoss(w_log, x_train, y_train, 1)
        # print(cost)
        # print(dw_log)
        w_log = w_log - learning_rate * dw_log
        # cost = np.linalg.norm(x_train@w_sl - y_train)

        if i % 100000 == 0:
            costs.append(cost)
            # Print the cost every 1000 training examples
        if 1 and i % 100000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    return w_log


w_log = model_logloss(X_train, Y_train, num_iterations=1000000, learning_rate=0.09)
# print(w_log)
w_s = SVM(X_train, Y_train)
w_log_norm = w_log/np.linalg.norm(w_log)
w_s = w_s/np.linalg.norm(w_s)
# print(w_log_norm)
# print(w_s)
distance2 = np.linalg.norm(w_log_norm[0]/w_log_norm[1]-w_s[0]/w_s[1])
print(distance2)

xp = np.linspace(np.min(X_train[:, 0]), np.max(X_train[:, 0]), 100)
yp = - (w_s[0] * xp) / w_s[1]
idx0 = np.where(Y_train == -1)
idx1 = np.where(Y_train == 1)

plt.plot(X_train[idx0, 0], X_train[idx0, 1], 'rx')
plt.plot(X_train[idx1, 0], X_train[idx1, 1], 'bo')
plt.plot(xp, yp, '--b', label='SVM')

yp1 = - (w_log[0] * xp) / w_log[1]
plt.plot(xp, yp1, '-r', label='log')
plt.title('log')
plt.legend()
plt.show()

