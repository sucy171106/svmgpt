import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression



# Generate a toy dataset
n_samples = 100
x_train, y = datasets.make_blobs(n_samples=n_samples, n_features=700, centers=2, cluster_std=[1.0, 1.0], random_state=42)
y = y * 2 - 1
y = np.reshape(y, (len(y), 1))

b = np.ones(n_samples)
n_features_list = np.arange(10, 710, 10)

distances = []

for n_features in n_features_list:
    x = x_train[:, :n_features]


    # LS
    xt = np.insert(x, 0, b, axis=1)
    w_1 = np.linalg.pinv(xt).dot(y).ravel()

    lr = LinearRegression()
    lr.fit(x, y)
    w_2 = np.insert(lr.coef_, 0, lr.intercept_, axis=1).ravel()

    # compute the distance
    distance = np.linalg.norm(w_1 - w_2)
    distances.append(distance)
fig, ax = plt.subplots()
ax.set_title('distance')
ax.plot(n_features_list, distances)
plt.show()