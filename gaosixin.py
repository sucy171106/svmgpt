import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

# Generate a toy dataset
n_samples = 50
x_train, y = datasets.make_blobs(n_samples=n_samples, n_features=700, centers=2, cluster_std=[1.0, 1.0], random_state=42)
y = y * 2 - 1
y = np.reshape(y, (len(y), 1))

b = np.ones(n_samples)
n_features_list = np.arange(5, 401, 5)

distances = []

for n_features in n_features_list:
    x = x_train[:, :n_features]


    # SVM
    n = x.shape[1]
    s_s = cp.Variable((n, 1))
    b_s = cp.Variable()

    objective = cp.Minimize(cp.norm(s_s) ** 2)
    constraints = [cp.multiply(y, x @ s_s + b_s) >= 1]
    prob = cp.Problem(objective, constraints)

    result = prob.solve()
    s_s_value = s_s.value.astype(np.int64)
    b_s_value = b_s.value.astype(np.int64)
    w_s = np.append(b_s_value, s_s_value).ravel()

    # LS
    xt = np.insert(x, 0, b, axis=1)
    w_l = np.linalg.pinv(xt).dot(y).ravel()

    #lr = LinearRegression()
    #lr.fit(x, y)
    #w_l = np.insert(lr.coef_, 0, lr.intercept_, axis=1).ravel()
    # compute the distance
    distance = np.linalg.norm(w_s - w_l)
    distances.append(distance)

fig, ax = plt.subplots()
ax.set_title('distance between w_s and w_l')
ax.plot(n_features_list/n_samples, np.log(distances)) #log
plt.show()