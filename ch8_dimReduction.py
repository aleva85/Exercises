"""
Created on:  Wed Apr 11 20:13:55 2018
Author:      Alessandro Nesti
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

np.random.seed(42)

m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)


# PCA identifies the axis in the dataset dimensional space that accounts for the largest amount of variance
# the unity vectors identifieng these axis are called principal components

# PCA using SVD
X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered)   # singular value decomposition 
c1 = Vt.T[:, 0]   # get first 2 principal components
c2 = Vt.T[:, 1]

m, n = X.shape
S = np.zeros(X_centered.shape)
S[:n, :n] = np.diag(s)

W2 = Vt.T[:, :2]
X2D_using_svd = X_centered.dot(W2)

X3D_inv_using_svd = X2D_using_svd.dot(Vt[:2, :]) # try reconstructing, some error

# in scikitlearn:
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X2D = pca.fit_transform(X)

X3D_inv = pca.inverse_transform(X2D) # try to retrieve original data, some info loss
pca.components_
pca.explained_variance_ratio_

# choosing the right number of components
pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1

# new dataset
from six.moves import urllib
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
from sklearn.model_selection import train_test_split
X = mnist["data"]
y = mnist["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y)

# incremental PCA - split training set in mini batches, good for big datasets
from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X, n_batches):
    print(".", end="") # not shown in the book
    inc_pca.partial_fit(X_batch)

X_reduced = inc_pca.transform(X_train)

# randomized PCA - stochastic, quickly find an approximation of the first d pc 
rnd_pca = PCA(n_components=154, svd_solver="randomized", random_state=42)
X_reduced = rnd_pca.fit_transform(X_train)

# kernel PCA - good for nonlinear datasets
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_swiss_roll

lin_pca = KernelPCA(n_components = 2, kernel="linear", fit_inverse_transform=True)
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
sig_pca = KernelPCA(n_components = 2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)

X_reduced = rbf_pca.fit_transform(X_train)

### select kernel and tune hyperparam
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
y = t > 6.9

clf = Pipeline([
        ("kpca", KernelPCA(n_components=2)),
        ("log_reg", LogisticRegression())
    ])

param_grid = [{
        "kpca__gamma": np.linspace(0.03, 0.05, 10),
        "kpca__kernel": ["rbf", "sigmoid"]
    }]

grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y)


# other dim reduction techniques
# LLE
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
X_reduced = lle.fit_transform(X)



