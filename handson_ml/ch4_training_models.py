"""
Created on:  Mon Apr  2 17:34:39 2018
Author:      Alessandro Nesti
"""

from handy_module import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
np.random.seed(42)      # to make this notebook's output stable across runs

# make synthetic data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# fit with normal equation
X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# using scikit code
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_

# batch gradient descent: compute partial derivatives using the whole batch of training data at every step
# stochastic gradient descent - pick a random instance and compute gradient on that instance. 
# one way to converge is reducing the learning rate over time
# Mini-batch gradient descet compute gradient on small random set of training instances

# examples (learning rate eta)
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1, random_state=42)
sgd_reg.fit(X, y.ravel())

# Polynomial regression: add powers of each feture as new features
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)


# possible to asses if a model has the proper complexity by looking at the learning curves:
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)   # not shown in the book
    plt.xlabel("Training set size", fontsize=14) # not shown
    plt.ylabel("RMSE", fontsize=14)              # not shown
    
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)
plt.axis([0, 80, 0, 3])                         # not shown in the book
#save_fig("underfitting_learning_curves_plot")   # not shown
plt.show()                                      # not shown
# underfitting model! because error is high

from sklearn.pipeline import Pipeline
polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
        ("lin_reg", LinearRegression()),
    ])
plot_learning_curves(polynomial_regression, X, y)
plt.axis([0, 80, 0, 3])           # not shown
#save_fig("learning_curves_plot")  # not shown
plt.show()                        # not shown


# regularization!
# NB: should only regularize during trainig
# NB: scale the data (StandardScaler() ) before regularizing
# look at forluas for immediate understanding

# Ridge regression: keep weights small large alpha-> highly regularized
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
ridge_reg.fit(X, y)

sgd_reg = SGDRegressor(max_iter=5, penalty="l2", random_state=42)
sgd_reg.fit(X, y.ravel())
# specifying l2 in sgdr means doing ridge regularization

# Lasso regression - like ridge, but uses l1 norm rather than l2
# eliminates weights of least important features, ie, does automatic feature selection
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)

# elastic net - does a bit of both lasso and ridge, depending on settings
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_net.fit(X, y)

# another way to regularize is to stop learning earlier (early stopping)
# works because error on test set tends to increase if training over too many epochs (overfitting)

#####
# do logistic regression on the iris
from sklearn import datasets
iris = datasets.load_iris()
list(iris.keys())

X = iris["data"][:, 3:]  # petal width
y = (iris["target"] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X, y)

X_new = [[1.7], [1.5]]
y_proba = log_reg.predict_proba(X_new)
pred = log_reg.predict(X_new)

# now use 2 features
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.int)

log_reg = LogisticRegression(C=10**10, random_state=42)
log_reg.fit(X, y)

X_new = [[5.5,1.7], [5,1.5]]
y_proba = log_reg.predict_proba(X_new)
pred = log_reg.predict(X_new)
# could be regularized : high C param, less regularization
# could classify multiple classes directly, using softmax reg
    # compute a softmax score for every class, for the instance x
    # run scores trough softmax function to get probabilities of belonging to each class
    
y = iris["target"]

softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10, random_state=42)
softmax_reg.fit(X, y)

