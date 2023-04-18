import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn import datasets
from sklearn.metrics import zero_one_loss
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Generate a toy dataset
n_samples = 50
#**********dim = 2 or 400
dim = 200
# X, Y = make_classification(n_samples=n_samples, n_features=dim, n_redundant=0, n_clusters_per_class=1, class_sep=2, random_state=42)

X, Y = datasets.make_blobs(n_samples=n_samples, n_features=dim, centers=2, cluster_std=[3.0, 3.0], random_state=42)
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

X, X_test, Y_train, Y_test = train_test_split(X, Y_train, test_size=0.2, random_state=52)

n_features_list = np.arange(2, dim , 1)
distances1 = []
distances2= []

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


#dim = 2 for check
if (dim == 2):
    w_s = SVM(X_train, Y_train)

    xp = np.linspace(np.min(X_train[:, 0]), np.max(X_train[:, 0]), 100)
    yp = - (w_s[0] * xp) / w_s[1]
    idx0 = np.where(Y_train == -1)
    idx1 = np.where(Y_train == 1)

    plt.plot(X_train[idx0, 0], X_train[idx0, 1], 'rx')
    plt.plot(X_train[idx1, 0], X_train[idx1, 1], 'bo')
    plt.plot(xp, yp, '--b', label='SVM')
    plt.legend()
    plt.show()
#high dim for verify task one
else:
    distances1 = []
    for n_features in n_features_list:
        #print(n_features_list)
        X_train = X[:, :n_features]
        x_test = X_test[:, :n_features]
        # {-1,1} label for log
        poly = PolynomialFeatures(2)
        poly.fit_transform(X)
        w_s = SVM(X_train, Y_train)
        # print(w_log)
        Y_pre = np.sign(X_train @ w_s)
        print(Y_pre)
        print(Y_train)
        trainrisk = zero_one_loss(Y_pre, Y_train)
        distances1.append(trainrisk)

        Y_pre =np.sign(x_test @ w_s)
        exprisk = np.mean(zero_one_loss(Y_pre, Y_test))
        distances2.append(exprisk)

    print(distances1)
    plt.plot(n_features_list / n_samples, distances1,'x')
    plt.plot(n_features_list / n_samples, distances2)
    plt.show()
