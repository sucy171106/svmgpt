import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn import datasets
from sklearn.metrics import zero_one_loss
from sklearn.metrics import mean_squared_error
# Generate a toy dataset
n_samples = 24
dim = 2
X, Y = datasets.make_blobs(n_samples=n_samples, n_features=dim, centers=2, cluster_std=[2.0, 2.0], random_state=42)
Y = np.reshape(Y, (len(Y), 1))
Y_train = Y * 2 - 1

X_train = X

distances2 = []


def SVM(x_train, y_train):
    # SVM
    n = x_train.shape[1]
    s_s = cp.Variable((n, 1))
    # b_s = cp.Variable()

    objective = cp.Minimize(cp.norm(s_s) ** 2)
    constraints = [cp.multiply(y_train, x_train @ s_s) >= 1]
    prob = cp.Problem(objective, constraints)

    prob.solve()
    s_s_value = s_s.value
    # b_s_value = b_s.value
    return s_s_value

# GRADED FUNCTION: sigmoid
def sigmoid(x):
    if np.all(x >= 0):  # 对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))

'''
def model_logloss(x_train, y_train, num_iterations=20000, learning_rate=0.001):
    s_log = np.zeros((x_train.shape[0], 1))  # GRADED FUNCTION: initialize_with_zeros
    b = 0
    costs = []

    # Gradient descent
    # GRADED FUNCTION: optimize
    for i in range(num_iterations):
        # GRADED FUNCTION: propagate
        m = x_train.shape[1]
        A = sigmoid(np.dot(s_log.T, x_train) + b)
        cost = -(1.0 / m) * np.sum(y_train * np.log(A) + (1 - y_train) * np.log(1 - A))
        ds_log = (1.0 / m) * np.dot(x_train, (A - y_train).T)
        db = (1.0 / m) * np.sum(A - y_train)

        # update rule
        cost = np.squeeze(cost)
        s_log = s_log - learning_rate * ds_log
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
        # Print the cost every 1000 training examples
        if 1 and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    
    return w_log
'''
def l(x):

    if np.all(x >= 0):  # 对sigmoid函数的优化，避免了出现极大的数据溢出
        return np.log(1+np.exp(-x))
    else:
        return np.log(1+np.exp(x))+np.exp(x)

def model_logloss(x_train, y_train, num_iterations=20000, learning_rate=0.01):
    s_log = np.zeros((x_train.shape[1], 1))  # GRADED FUNCTION: initialize_with_zeros
    costs = []

    # Gradient descent
    # GRADED FUNCTION: optimize
    for i in range(num_iterations):
        # GRADED FUNCTION: propagate
        m = x_train.shape[0]
        cost = np.mean(l(y_train.T@x_train@s_log))

        A = sigmoid(np.dot(x_train, s_log))
        ds_log = np.dot(x_train.T, (A - y_train))/m

        y_pred = np.dot(x_train, s_log)
        #ds_log = -y_train.T * x_train / (1 + np.exp(y_train.T * y_pred))
        ds_log = (np.mean(l(y_train.T@x_train@(s_log+0.000000001))) - np.mean(l(y_train.T@x_train@(s_log))) )/0.000000001
        # update rule
        cost = np.squeeze(cost)
        s_log = s_log - learning_rate * ds_log

        if i % 50 == 0:
            costs.append(cost)
            # Print the cost every 1000 training examples
            print("Cost after iteration %i: %f" % (i, cost))
            w_log = np.flipud(s_log)
            w_loghat= w_log / np.linalg.norm(w_log)
            print(w_loghat)

    return w_loghat


w_s = SVM(X_train, Y_train)

w_log = model_logloss(X_train, Y_train, num_iterations=2000, learning_rate=0.001)

distance1 = np.linalg.norm(w_s - w_log)

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