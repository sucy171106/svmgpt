import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Generate a random dataset
X, y = make_classification(n_samples=1000, n_features=100, n_informative=50, n_redundant=0, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM with varying amounts of training data
train_sizes = [100, 200, 400, 800]
test_errors = []
train_errors = []
for size in train_sizes:
    X_train_small = X_train[:size]
    y_train_small = y_train[:size]
    svm = SVC(kernel='linear')
    svm.fit(X_train_small, y_train_small)
    test_errors.append(1 - svm.score(X_test, y_test))
    train_errors.append(1 - svm.score(X_train_small, y_train_small))

# Plot the results
fig, ax = plt.subplots()
ax.plot(train_sizes, train_errors, label='Training error')
ax.plot(train_sizes, test_errors, label='Test error')
ax.set_xlabel('Number of training examples')
ax.set_ylabel('Error')
ax.set_title('Double Descent for SVM')
ax.legend()
plt.show()
