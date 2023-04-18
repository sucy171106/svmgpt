import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss


# Generate a toy dataset
x, y = datasets.make_blobs(n_samples=300, n_features=900, centers=2, cluster_std=[1.0, 1.0], random_state=42)
y = y * 2 - 1
y = np.reshape(y, (len(y), 1))
x_train, X_test, y, y_test = train_test_split(x, y, test_size=0.2, random_state=52)

n_features_list = np.arange(10, 910, 10)

train_errors_l = []
test_errors_l = []
test_01errors_l = []

train_errors_s = []
test_errors_s = []
test_01errors_s = []

distances = []

for n_features in n_features_list:
    x = x_train[:, :n_features]
    x_test = X_test[:, :n_features]

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

    w_s = np.append(s_s_value, b_s_value)

    # Compute the training and test errors
    train_error_s = mean_squared_error(y, x @ s_s_value + b_s_value)
    test_error_s = mean_squared_error(y_test, x_test @ s_s_value + b_s_value)
    test_01error_s = zero_one_loss(y_test, x_test @ s_s_value + b_s_value)

    # Store the training and test errors
    train_errors_s.append(train_error_s)
    test_errors_s.append(test_error_s)

    # LS
    w_l = np.linalg.pinv(x).dot(y)
    xp = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 100)
    yp = - (w_l[0] * xp) / w_l[1]

    # Compute the training and test errors
    train_error_l = mean_squared_error(y, x @ w_l)
    test_error_l = mean_squared_error(y_test, x_test @ w_l)
    test_01error_l = zero_one_loss(y_test, x_test @ w_l.astype(np.int64))

    # Store the training and test errors
    train_errors_l.append(train_error_l)
    test_errors_l.append(test_error_l)

    # compute the distance
    distance = np.linalg.norm(w_s - w_l)
    distances.append(distance)

    # compute the 0-1 loss
    test_01errors_l.append(test_01error_l)
    test_01errors_s.append(test_01error_s)

fig, ax = plt.subplots()
ax.set_title('distance')
ax.plot(n_features_list, distances)
ax.plot(n_features_list, test_01errors_s)
plt.show()