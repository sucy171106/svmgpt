
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import zero_one_loss
import cvxpy as cp
from matplotlib.colors import ListedColormap
from sklearn import datasets
kwargs = {'fit_intercept': False, 'dual': True, 'C': 64, 'tol': 1e-6, 'max_iter': 5000}
# Generate a toy dataset
# X, y = make_regression(n_samples=100, n_features=200, n_informative=100,noise=1.0, random_state=42)
dim = 2
sample= 20
# x_train, y = datasets.make_classification(n_samples=sample, n_features=dim, random_state=42)
# #Y = Y * 2 - 1
# y = np.reshape(y, (len(y), 1))
# r = np.ones(x_train.shape[0])
#x_train = np.insert(x_train, 0, r, axis=1)
rng = np.random.RandomState(1)
observed = rng.uniform(low=-1, high=1, size=(sample,dim))

print(observed)
labels = np.repeat([-1, 1], int((sample + 1) / 2))[:sample, None]  # drop last one if necessary
inputs = observed * labels
print(labels)
x= observed
y= labels
# Construct the problem.

print(y.shape)
n_features_list = np.arange(sample, dim+1, 10)


svm =  LinearSVC(**kwargs)
distances= []

idx0 = np.where(y == -1)
idx1 = np.where(y == 1)

plt.plot(x[idx0, 0], x[idx0, 1], 'rx')
plt.plot(x[idx1, 0], x[idx1, 1], 'bo')



plt.title('log')
plt.legend()
plt.show()

for n_features in n_features_list:
    x = x_train[:, :n_features]
    #x_test = X_test[:, :n_features]
    if n_features%100 ==0:
        print(n_features)
    n = x.shape[1]
    svm.fit(x_train,y.ravel())
    wsk = svm.coef_
    # s_LR = cp.Variable((n, 1))
    # objective1 = cp.Minimize(cp.norm(s_LR)** 2)
    # constraints1 = [y == x @ s_LR]
    # prob1 = cp.Problem(objective1, constraints1)
    # prob1.solve()
    # s_LR_value = s_LR.value
    # w_l = s_LR_value
    #print(w_l)
    # s_s = cp.Variable((n, 1))
    # objective = cp.Minimize(cp.norm(s_s) ** 2)
    # constraints = [cp.multiply(y, x @ s_s) >= 1]
    # prob = cp.Problem(objective, constraints)
    # result = prob.solve()
    # s_s_value = s_s.value
    # w_s = s_s_value
    #print(w_s)

    w_l2 = np.linalg.pinv(x).dot(y).ravel()

    #distance = np.linalg.norm(wsk/np.linalg.norm(wsk)-w_l/np.linalg.norm(w_l))
    #distances .append(distance)
#print(w_s[0:3])
#print(w_l[0:3])
print(wsk[0][0:3])
print(w_l2[0:3])


