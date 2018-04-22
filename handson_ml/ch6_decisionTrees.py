"""
Created on:  Sun Apr  8 09:54:23 2018
Author:      Alessandro Nesti
"""

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X, y)


# visualize with graphviz
# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "decision_trees"

def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id)

from sklearn.tree import export_graphviz

export_graphviz(
        tree_clf,
        out_file="iris_tree.pdf",
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )

# Regularization: restric decision tree freedom during training to avoid overfitting
# reduce max_depth
# min_sample_split - min num of samples to perform a split
# min_sample_leaf
# min_weight_fraction_leaf
# max_features

# regression
np.random.seed(42)
m = 200
X = np.random.rand(m, 1)
y = 4 * (X - 0.5) ** 2
y = y + np.random.randn(m, 1) / 10

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg.fit(X, y)





