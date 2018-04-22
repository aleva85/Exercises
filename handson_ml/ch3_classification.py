"""
Created on:   Mon Apr  2 11:33:33 2018
@author:      Alessandro Nesti
"""

from handy_module import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
np.random.seed(42)      # to make this notebook's output stable across runs

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
           interpolation="nearest")
plt.axis("off")
std_format_fig()

#plt.savefig('\ch3_fig', format='png', dpi=300)
#plt.savefig('test_fig')
#save_fig("some_digit_plot",path = '\ch3_fig')
plt.show()

# split train test, and ensure with shuffling that folders are nmot missing digits
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

### binary classifier: classify 5s
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# train
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
sgd_clf = SGDClassifier(max_iter=5, random_state=42)
sgd_clf.fit(X_train, y_train_5)

# thresholds
# standard decision threshold is 0, but I can influence performances (eg F1 scors) through this 
# for this, I can retrieve, always with cros_val_predict, decision scores, rather than decisions
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")

# validate 
# cross validation
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# not a good way to validate, only 10% are 5s

#confusion matrix
from sklearn.metrics import confusion_matrix
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
confusion_matrix(y_train_5, y_train_pred)
# perfect classifier only has nonzero values on the diagonal

# precision-recall
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
precision_score(y_train_5, y_train_pred) # of all my yes, how many were right
recall_score(y_train_5, y_train_pred)    # of all the yes, how many I cought
f1_score(y_train_5, y_train_pred)        # harmonic mean (penalize small values) of prec and recall. high scores to classifiers with similar precisions and recalls

# precision-recall - choose the right threshold
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])
plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-700000, 700000])
plt.show()

#precision-recall - plot one vs the other
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
#save_fig("precision_vs_recall_plot")
plt.show()

# ROC 
# as for prec-rec, use the function to retrieve values of false positives and true positives for different th scores
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
#save_fig("roc_curve_plot")
plt.show()


# for fun, lets do the same for a forest
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")

y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)




# Multiclass classification
# some classifiers can directly handle multiple classes (random foresty, naive bayes)
# others are strictly binary, but can use different strategies

# eg, train 1 binary classifier for every class (One vs All). in scikitlearn, this is the standard for all except SVM 
# alternatively, train one class vs another for all pair of classes, and count which class won the most duels

sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])

# the decision function now returns 10 scores per each class
some_digit_scores = sgd_clf.decision_function([some_digit])

# to do ovo, you have to ask explicitly
from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(max_iter=5, random_state=42))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit])

# try random forest:
forest_clf.fit(X_train, y_train)
forest_clf.predict([some_digit])
forest_clf.predict_proba([some_digit])

# finasl note, don't forget to scale features! in this case improves from 84% to 91%!
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy") # evaluate performance
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

