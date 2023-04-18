import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn import datasets
from sklearn.metrics import zero_one_loss
from sklearn.metrics import mean_squared_error
# Generate a toy dataset
n_samples = 50
dim = 2
X, Y = datasets.make_blobs(n_samples=n_samples, n_features=dim, centers=2, cluster_std=[2.0, 2.0], random_state=42)
Y = np.reshape(Y, (len(Y), 1))
Y_train = Y * 2 - 1


r = np.ones(X.shape[0])


n_features_list = np.arange(2, dim+1, 20)
distances1 = []
distances2 = []


def SVM(x_train, y_train):
    # SVM
    s_s = cp.Variable(( x_train.shape[1], 1))


    objective = cp.Minimize(cp.norm(s_s) ** 2)
    constraints = [cp.multiply(y_train, x_train @ s_s ) >= 1]
    prob = cp.Problem(objective, constraints)
    print("result")
    result = prob.solve()
    s_s_value = s_s.value
    print(s_s_value)
    return s_s_value


# GRADED FUNCTION: sigmoid
def sigmoid(inX):
    return  1 / (1 + np.exp(inX))


def model_logloss(x_train, y_train, num_iterations=20000, learning_rate=0.001):
    s_log = np.zeros((x_train.shape[0], 1))  # GRADED FUNCTION: initialize_with_zeros
    b = 0
    costs = []

    # Gradient descent
    # GRADED FUNCTION: optimize
    for i in range(num_iterations):
        # GRADED FUNCTION: propagate
        m = x_train.shape[1]
        A = sigmoid(np.dot(s_log.T, x_train))
        ds_log = (1.0 / m) * np.dot(x_train, (A - y_train).T)
        # update rule
        s_log = s_log - learning_rate * ds_log

    return s_log/np.linalg.norm(s_log)
'''


def model_logloss(x_train, y_train, num_iterations=20000, learning_rate=0.001):
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

'''

for n_features in n_features_list:
    X_train = X[:, :n_features]
    w_log = model_logloss(X_train.T, Y_train.T, num_iterations=50000, learning_rate=0.0006)
    #w_log = model_logloss(X_train, Y_train, num_iterations=50000, learning_rate=0.00006)
    w_s = SVM(X_train, Y_train)

    distance1 = np.linalg.norm(w_s - w_log)
    distances1.append(distance1)

xp = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
yp = - (w_s[0] * xp ) / w_s[1]
yp1 = np.array(- (w_s[0] * xp - 1) / w_s[1])  # margin boundary for support vectors for y=1
yp0 = np.array(- (w_s[0] * xp + 1) / w_s[1])  # margin boundary for support vectors for y=0

idx0 = np.where(Y_train == -1)
idx1 = np.where(Y_train == 1)


plt.plot(X[idx0, 0], X[idx0, 1], 'rx')
plt.plot(X[idx1, 0], X[idx1, 1], 'bo')
plt.plot(xp, yp, '-y',label='SVM')
plt.plot(xp, yp1, '--g', xp, yp0, '--r')
plt.title('decision boundary for a linear SVM classifier and LS'.format(1))


#plt.figure(1)
#plt.plot(n_features_list / n_samples, distances1)
plt.show()