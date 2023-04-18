import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn import datasets
# Generate a toy dataset
n_samples = 50
dim = 400
X, Y = datasets.make_blobs(n_samples=n_samples, n_features=dim, centers=2, cluster_std=[1.0, 1.0], random_state=42)
Y = Y * 2 - 1
Y_train = np.reshape(Y, (len(Y), 1))


r = np.ones(X.shape[0])
X_train = np.insert(X, 0, r, axis=1)

print(X_train.shape)
print(Y_train.shape)
n_features_list = np.arange(5, dim+1, 5)
distances1 = []
distances2 = []


def SVM(x_train, y_train):
    # SVM
    n = x_train.shape[1]
    s_s = cp.Variable((n, 1))
    b_s = cp.Variable()

    objective = cp.Minimize(cp.norm(s_s) ** 2)
    constraints = [cp.multiply(y_train, x_train @ s_s + b_s) >= 1]
    prob = cp.Problem(objective, constraints)

    result = prob.solve()
    s_s_value = s_s.value.astype(np.int64)
    b_s_value = b_s.value.astype(np.int64)
    w_s = np.append(b_s_value, s_s_value).ravel()
    return w_s


def leastsquare(x_train, y_train):
    # LS
    xt = np.insert(x_train, 0, y_train, axis=1)
    w_l = np.linalg.pinv(xt).dot(y_train).ravel()
    return w_l


# GRADED FUNCTION: model
def model_squreloss(x_train, y_train, num_iterations=2000, learning_rate=0.5):
    w = np.zeros((x_train.shape[1], 1)) # GRADED FUNCTION: initialize_with_zeros
    # Gradient descent
    # GRADED FUNCTION: optimize
    for i in range(num_iterations):

        # GRADED FUNCTION: propagate
        dw_sl = np.dot(x_train.T, (np.dot(x_train, w) - y_train))
        # update rule
        w_sl = w - learning_rate * dw_sl
    return w_sl, dw_sl


# GRADED FUNCTION: sigmoid
def sigmoid(f):
    s = 1.0 / (1.0 + np.exp(-1.0 * f))
    return s


def model_logloss(x_train, y_train, num_iterations=2000, learning_rate=0.5):
    w = np.zeros((x_train.shape[1], 1)) # GRADED FUNCTION: initialize_with_zeros
    # Gradient descent
    # GRADED FUNCTION: optimize
    for i in range(num_iterations):

        # GRADED FUNCTION: propagate
        A = sigmoid(np.dot(x_train, w))
        dw_log = (1.0/x_train.shape[1])*np.dot(x_train.T, (A-y_train))
        # update rule
        w_log = w - learning_rate * dw_log
    return w_log, dw_log


for n_features in n_features_list:
    X_train = X[:, :n_features]
    w_sl, dw_sl = model_squreloss(X_train, Y_train, num_iterations=2000, learning_rate=0.005)
    w_log, dw_log = model_squreloss(X_train, Y_train, num_iterations=2000, learning_rate=0.005)
    w_l = leastsquare(X_train, Y_train)
    w_s = SVM(X_train, Y_train)
    distance1 = np.linalg.norm(w_s - w_log)
    distances1.append(distance1)
    distance2 = np.linalg.norm(w_sl - w_l)
    distances2.append(distance2)

plt.figure(1)
fig1 = plt.subplot(121)
fig1.set_title('distance between w_s and w_l')
plt.plot(n_features_list / n_samples, np.log(distances1))
fig2 = plt.subplot(122)
plt.plot(n_features_list / n_samples, np.log(distances2))
plt.show()