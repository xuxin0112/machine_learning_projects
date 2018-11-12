from sklearn.datasets import make_regression, make_friedman1, make_classification, make_blobs
from matplotlib import pyplot as plt
plt.figure(1)
plt.title('sample regression problem with one input variable')
X_R1, y_R1 = make_regression(n_samples=100, n_features=1, n_informative=1, bias = 150.0, noise=30, random_state=1)
plt.scatter(X_R1, y_R1, marker='o', s=50)

plt.figure(2)
plt.title('Complex regression problem with one input variable')
X_F1, y_F1 = make_friedman1(n_samples=100, n_features=7, random_state=0)
plt.scatter(X_F1[:, 2], y_F1, marker='o', s=50)

plt.figure(3)
plt.title('Sample binary classification problem with two informative features')
X_C1, y_C1 = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2,
                                 n_clusters_per_class=1, flip_y=0.1, class_sep=0.5, random_state=0)
plt.scatter(X_C1[:, 0], X_C1[:, 1], c=y_C1, s=50)


plt.figure(4)
X_D2, y_D2 = make_blobs(n_samples=100, n_features=2, centers=8, cluster_std=1.3, random_state=4)
y_D2 = y_D2 % 2
plt.title('Sample binary classification problem with non-linearly separable classes')
plt.scatter(X_D2[:, 0], X_D2[:, 1], c=y_D2, marker='o', s=50)
plt.show()