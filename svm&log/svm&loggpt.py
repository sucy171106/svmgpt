import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Generate separable data
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, class_sep=2.0, random_state=42)
y[y == 0] = -1

# Fit hard-margin SVM
svm = LinearSVC(C=1000)
svm.fit(X, y)


# Fit logistic regression using gradient descent
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logistic_loss(w, X, y):
    z = np.dot(X, w)
    return np.mean(np.log(1 + np.exp(-y * z)))


def gradient(w, X, y):
    z = np.dot(X, w)
    s = sigmoid(-y * z)
    return np.mean(-y[:, None] * X * s[:, None], axis=0)


w = np.zeros(X.shape[1])
alpha = 0.1
for i in range(10000):
    grad = gradient(w, X, y)
    w -= alpha * grad
    if i % 1000 == 0:
        print(f"Iteration {i}: loss={logistic_loss(w, X, y)}")

# Compare decision boundaries
print("SVM coef:", svm.coef_)
print("Logistic coef:", w)
