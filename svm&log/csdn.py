import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn import datasets
from sklearn.metrics import zero_one_loss
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_classification
# Generate a toy dataset
n_samples = 100
dim = 2
#X, Y = make_classification(n_samples=n_samples, n_features=dim, n_redundant=0, n_clusters_per_class=1, class_sep=2, random_state=42)

X, Y = datasets.make_blobs(n_samples=n_samples, n_features=dim, centers=2, cluster_std=[1.0, 1.0], random_state=42)
Y_train = np.reshape(Y, (len(Y), 1))
Y_train[Y_train== 0] = -1
X_train = X

r = np.ones(X.shape[0])
Xb_train = np.insert(X, 0, r, axis=1)


n_features_list = np.arange(5, dim+1, 20)
distances1 = []
distances2 = []


def SVM(x_train, y_train):
    # SVM
    n = x_train.shape[1]
    s_s = cp.Variable((n, 1))
    b_s = cp.Variable()

    objective = cp.Minimize(cp.norm(s_s) ** 2)
    constraints = [cp.multiply(y_train, x_train @ s_s ) >= 1]
    prob = cp.Problem(objective, constraints)

    result = prob.solve()
    s_s_value = s_s.value

    return s_s_value


def sigmoid(x):
    return  1. / (1 + np.exp(-x))
def model_logloss(x_train, y_train, num_iterations=200000, learning_rate=0):
    w_log = np.zeros((x_train.shape[1], 1))  # GRADED FUNCTION: initialize_with_zeros
    # Gradient descent
    # GRADED FUNCTION: optimize
    for i in range(num_iterations):
        # GRADED FUNCTION: propagate
        A = sigmoid(np.dot(x_train, w_log))
        dw_log = np.dot(x_train.T,A-y_train)/x_train.shape[0]
        #print(dw_log.shape)
        w_log = w_log - learning_rate * dw_log
    return w_log/np.linalg.norm(w_log)

w_s= SVM(X_train, Y_train)

print(w_s)
idx0 = np.where(Y == -1)
idx1 = np.where(Y == 1)
w_log = model_logloss(X_train, abs(Y_train), num_iterations=50000, learning_rate=0.001)

plt.plot(X_train[idx0, 0], X_train[idx0, 1], 'rx')
plt.plot(X_train[idx1, 0], X_train[idx1, 1], 'bo')

xp = np.linspace(np.min(X_train[:, 0]), np.max(X_train[:, 0]), 100)

yp = - (w_log[0] * xp ) / w_log[1]
plt.plot(xp, yp, '-y',label='log')

yp1 = - (w_s[0] * xp ) / w_s[1]
plt.plot(xp, yp1, '--p',label='SVM')

plt.title('decision boundary for a linear SVM classifier and LS')
plt.legend()
plt.show()