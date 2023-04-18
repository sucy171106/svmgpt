from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import zero_one_loss
from sklearn import datasets

# Problem data.



# Load MNIST dataset
mnist = fetch_openml('mnist_784')
X, y = mnist.data, mnist.target.astype(int)



# Reduce dataset size to 1/8
X, _, y, _ = train_test_split(X, y, test_size=0.875, random_state=42)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define SVM pipeline
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='poly', degree=20, gamma='auto'))
])

# Train SVM model
#svm_pipeline.fit(X_train, y_train)

# Evaluate model performance
#train_score = svm_pipeline.score(X_train, y_train)
#test_score = svm_pipeline.score(X_test, y_test)

#print(f'Train score: {train_score:.3f}')
#print(f'Test score: {test_score:.3f}')

# Plot double descent curve
n_samples = len(X_train)
test_errors = []

for i in range(2, n_samples , 500):
    X_train_subset, y_train_subset = X_train[:i], y_train[:i]
    svm_pipeline.fit(X_train_subset, y_train_subset)
    y_pred = svm_pipeline.predict(X_test)

    print(i)

    test_error = zero_one_loss(y_test, y_pred)
    test_errors.append(test_error)

#plt.plot(np.arange(1, n_samples + 1, 500), train_scores, 'o-', label='train')
#plt.plot(np.arange(1, n_samples + 1, 500), test_scores, 'o-', label='test')
plt.plot(np.arange(2, n_samples , 500), test_errors, 'o-', label='test')
plt.legend()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.show()