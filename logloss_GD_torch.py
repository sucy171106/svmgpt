import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import cvxpy as cp
import torch
# Generate a toy dataset
n_samples = 10
dim = 20
n = 100 # Number of instances
m = 10  # Number of Features

X = np.random.rand(n,m)
y = np.random.rand(n)
y = np.random.rand(n)
ybin = [(int(yi >= 0.5) - int(yi < 0.5)) for yi in y]
y = np.array(ybin)
w = np.random.rand(m, 1)
print(y)
print(X)

print(X.shape)
n_features_list = np.arange(3, dim+1, 10)
costs =[]
distances2 = []

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


# GRADED FUNCTION: model
def model_squreloss(x_train, y_train, num_iterations=2000, learning_rate=0.5):

    w_sl = torch.zeros((x_train.shape[1], 1),requires_grad=True) # GRADED FUNCTION: initialize_with_zeros
    h_hat = torch.dot(w_sl,-torch.mm(x_train.T,y_train))
    loss = torch.sum(torch.log(1+torch.exp(h_hat)))
    # Gradient descent
    # GRADED FUNCTION: optimize
    for i in range(num_iterations):
        # GRADED FUNCTION: propagate
        #dw_sl = np.dot(x_train.T, (np.dot(x_train, w_sl) - y_train))
        w_sl.grad.zero_()
        loss.backward()
        dw_sl = torch.sum(w_sl.grad)
        # update rule
        w_sl = w_sl - learning_rate * dw_sl
        cost = np.linalg.norm(x_train@w_sl - y_train)
        if i % 10000 == 0:
            costs.append(cost)
            # Print the cost every 1000 training examples
        if 1 and i % 10000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    return w_sl.ravel()



w_sl = model_squreloss(X, Y_train, num_iterations=400000, learning_rate=0.00001)

w_s = SVM(X, Y_train)
distance2 = np.linalg.norm(w_sl - w_s)

print(w_sl)
print(w_s)
print(distance2)

