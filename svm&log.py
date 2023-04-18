import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn import datasets
from sklearn.metrics import zero_one_loss
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_classification

# Generate a toy dataset
n_samples = 24
#**********dim = 2 or 400
dim = 200
# X, Y = make_classification(n_samples=n_samples, n_features=dim, n_redundant=0, n_clusters_per_class=1, class_sep=2, random_state=42)

X, Y = datasets.make_blobs(n_samples=n_samples, n_features=dim, centers=2, cluster_std=[1.0, 1.0], random_state=42)
'''

Y_train = np.reshape(Y, (len(Y), 1))
Y_train[Y_train== 0] = -1

'''

#discount = 0.5
Y = np.reshape(Y, (len(Y), 1))
Y_train = Y * 2 - 1
X_train = X

r = np.ones(X.shape[0])
Xb_train = np.insert(X, 0, r, axis=1)

n_features_list = np.arange(5, dim + 1, 1)
distances1 = []


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

def l(x):
    return np.log(1+np.exp(-x))


def model_logloss(x_train, y_train, num_iterations=20000, learning_rate=0.001):
    w_log = np.zeros((x_train.shape[0], 1))  # GRADED FUNCTION: initialize_with_zeros
    grad = np.zeros((x_train.shape[0], 1))
    costs = []

    # Gradient descent
    # GRADED FUNCTION: optimize
    for i in range(num_iterations):
        # GRADED FUNCTION: propagate
        m = x_train.shape[1]
        A = sigmoid(np.dot(w_log.T, x_train))

        cost =  np.mean(l(y_train@x_train.T@w_log))
        dw_log = (1.0 / m) * np.dot(x_train, (A - y_train).T)
        '''
        grad = (grad * discount + learning_rate )* dw_log
        # update rule
        cost = np.squeeze(cost)
        w_log = w_log - grad
        '''
        # update rule
        cost = np.squeeze(cost)
        w_log = w_log - learning_rate* dw_log
        if i % 100 == 0:
            costs.append(cost)
        # Print the cost every 1000 training examples
        if 1 and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    return  np.flipud(w_log)

#dim = 2 for check
if (dim == 2):
    w_s = SVM(X_train, Y_train)
    w_log = model_logloss(X_train.T, Y_train.T, num_iterations=200, learning_rate=0.0001)

    xp = np.linspace(np.min(X_train[:, 0]), np.max(X_train[:, 0]), 100)
    yp = - (w_s[0] * xp) / w_s[1]
    idx0 = np.where(Y_train == -1)
    idx1 = np.where(Y_train == 1)

    plt.plot(X_train[idx0, 0], X_train[idx0, 1], 'rx')
    plt.plot(X_train[idx1, 0], X_train[idx1, 1], 'bo')
    plt.plot(xp, yp, '--b', label='SVM')

    yp1= - (w_log[0] * xp) / w_log[1]
    plt.plot(xp, yp1, '-r', label='log')
    plt.title('log')
    plt.legend()
    plt.show()
#high dim for verify task one
else:
    distances1 = []
    for n_features in n_features_list:
        print(n_features_list)
        X_train = X[:, :n_features]
        # {-1,1} label for log
        w_log = model_logloss(X_train.T,Y_train.T, num_iterations=200, learning_rate=0.0000001) #

        w_s = SVM(X_train, Y_train)
        # print(w_log)

        distance1 = np.linalg.norm(w_s  - w_log / np.linalg.norm(w_log))
        distances1.append(distance1)

    print(distances1)
    plt.plot(n_features_list / n_samples, distances1)
    plt.show()
