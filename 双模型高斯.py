import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
def SVM(x_train, y_train):
    # SVM
    n = x_train.shape[1]
    s_s = cp.Variable((n, 1))
    b_s = cp.Variable()

    objective = cp.Minimize(cp.norm(s_s) ** 2)
    constraints = [cp.multiply(y_train, x_train @ s_s + b_s) >= 1]
    prob = cp.Problem(objective, constraints)

    result = prob.solve()
    s_s_value = s_s.value
    b_s_value = b_s.value
    print(b_s_value)
    w_s = np.append(b_s_value, s_s_value).ravel()
    return w_s


# GRADED FUNCTION: sigmoid
def sigmoid(f):
    s = 1.0 / (1.0 + np.exp(-1.0 * f))
    return s


def model_logloss(x_train, y_train, num_iterations=20000, learning_rate=0.001):
    w_log = np.zeros((x_train.shape[1], 1))  # GRADED FUNCTION: initialize_with_zeros
    b = 0
    print(w_log.shape)
    costs = []
    # Gradient descent
    # GRADED FUNCTION: optimize
    for i in range(num_iterations):
        # GRADED FUNCTION: propagate
        m = x_train.shape[0]
        A = sigmoid(np.dot(x_train, w_log) + b)
        cost = -(1.0 / m) * np.sum(y_train * np.log(A) + (1 - y_train) * np.log(1 - A))

        dw_log = (1.0 / m) * np.dot(x_train.T, (A - y_train))
        db = (1.0 / m) * np.sum(A - y_train)
        # update rule
        cost = np.squeeze(cost)
        w_log = w_log - learning_rate * dw_log
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
        # Print the cost every 1000 training examples
        if 1 and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    return w_log, b

# Problem data.
x, y = datasets.make_blobs(n_samples=200, n_features=2, centers=2, cluster_std=[2.0, 2.0], random_state=42)
y = y * 2 - 1
y = np.reshape(y, (len(y), 1))


n = x.shape[1]
print(n)
C = 1

# SVM
s = cp.Variable((n, 1))
b = cp.Variable()

objective = cp.Minimize(cp.norm(s) ** 2)
constraints = [cp.multiply(y, x @ s + b) >= 1]
prob = cp.Problem(objective, constraints)

# The optimal objective value is returned by `prob.solve()`.
# The optimal value for x is stored in `x.value`.
result = prob.solve()

xp = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 100)
yp = - (s.value[0] * xp + b.value) / s.value[1]
yp1 = np.array(- (s.value[0] * xp + b.value - C) / s.value[1])  # margin boundary for support vectors for y=1
yp0 = np.array(- (s.value[0] * xp + b.value + C) / s.value[1])  # margin boundary for support vectors for y=0

idx0 = np.where(y == -1)
idx1 = np.where(y == 1)

print(s.value)
print(b.value)
plt.plot(x[idx0, 0], x[idx0, 1], 'rx')
plt.plot(x[idx1, 0], x[idx1, 1], 'bo')
plt.plot(xp, yp, '-y',label='SVM')
plt.plot(xp, yp1, '--g', xp, yp0, '--r')
plt.title('decision boundary for a linear SVM classifier and LS'.format(C))

w_s= SVM(x, y)
print(w_s)


w_log, b= model_logloss(x, y, num_iterations=2000, learning_rate=0.000007)
print(w_log)
print(b)

clf = LogisticRegression(random_state=0)
clf.fit(x,y.ravel())
#------打印结果------------------------
print("模型参数："+str(clf.coef_))
print("模型阈值："+str(clf.intercept_))

# Construct the LS
b = np.ones(200)
xt = np.insert(x,0,b,axis=1)
w = np.linalg.pinv(xt).dot(y)
yp = - (w[1]*xp+w[0])/w[2]

# 优化和回归的区别
idx0 = np.where(y == -1)
idx1 = np.where(y == 1)

plt.plot(xp, yp,label='LS')
plt.legend()
plt.show()