import numpy as np
from sklearn import datasets
from sklearn.svm import SVC

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # We only take the first two features.
y = iris.target

# Split the dataset into training and testing sets
n_samples = len(X)
n_train = int(0.8 * n_samples)
idx = np.random.permutation(n_samples)
train_idx, test_idx = idx[:n_train], idx[n_train:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Train SVM with different regularization strengths
alphas = np.logspace(-4, 4, 100)
train_accs, test_accs = [], []
for alpha in alphas:
    clf = SVC(kernel='linear', C=alpha)
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

# Plot the results
import matplotlib.pyplot as plt
plt.plot(alphas, train_accs, label='Train')
plt.plot(alphas, test_accs, label='Test')
plt.xscale('log')
plt.xlabel('Regularization strength (C)')
plt.ylabel('Accuracy')
plt.title('Double Descent for SVM')
plt.legend()
plt.show()