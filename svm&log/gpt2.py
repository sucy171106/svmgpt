import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Generate linearly separable data
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_clusters_per_class=1, class_sep=2, random_state=42)

# Fit logistic regression with gradient descent
lr = LogisticRegression(penalty='none', solver='saga', max_iter=10000)
lr.fit(X, y)

# Fit hard-margin SVM
svm = LinearSVC(C=1e10, max_iter=10000)
svm.fit(X, y)

