from sklearn.tree import DecisionTreeClassifier, plot_tree
import numpy as np
from DEPENDENCIES.constants import *
import DEPENDENCIES.processing as proc
import matplotlib.pyplot as plt

def find_best_tree(X_set, Y_set, max_depth=2):
    metric = 0
    for i in range(1000):
        dt = DecisionTreeClassifier(max_depth=max_depth, random_state=i, splitter='random')
        dt.fit(X_set, Y_set)
        Y_pred = dt.predict(X_set)
        tp, tn, fp, fn = proc.confusion_matrix_sfu(Y_set, Y_pred)
        current_metric = (tp+tn)/(tp+tn+fp+fn) #Quality metric. Accuracy
        if current_metric > metric:
            best_state = i*1
            metric = current_metric*1
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=best_state, splitter='random')
    dt.fit(X_set, Y_set)
    Y_pred = dt.predict(X_set)
    tp, tn, fp, fn = proc.confusion_matrix_sfu(Y_set, Y_pred)
    print("Accuracy: {:.2f}".format((tp+tn)/(tp+tn+fp+fn)))
    print("Sensitivity: {:.2f}".format(tp/(tp+fn)))
    print("Specificity: {:.2f}".format(tn/(tn+fp)))
    return dt

def find_best_tree_with_score(X_set, Y_set, max_depth=2):
    metric = 0
    for i in range(1000):
        dt = DecisionTreeClassifier(max_depth=max_depth, random_state=i, splitter='random')
        dt.fit(X_set, Y_set)
        Y_pred = dt.predict(X_set)
        tp, tn, fp, fn = proc.confusion_matrix_sfu(Y_set, Y_pred)
        current_metric = (tp+tn)/(tp+tn+fp+fn) #Quality metric. Accuracy
        if current_metric > metric and 'Score' in X_set.columns[np.argsort(dt.feature_importances_)[-1]]:
            best_state = i*1
            metric = current_metric*1
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=best_state, splitter='random')
    dt.fit(X_set, Y_set)
    Y_pred = dt.predict(X_set)
    tp, tn, fp, fn = proc.confusion_matrix_sfu(Y_set, Y_pred)
    print("Accuracy: {:.2f}".format((tp+tn)/(tp+tn+fp+fn)))
    print("Sensitivity: {:.2f}".format(tp/(tp+fp)))
    print("Specificity: {:.2f}".format(tn/(tn+fn)))
    return dt

def plot_my_tree(tree, ml_cols):
    fig = plt.figure(figsize=(18,8))
    ax = plt.axes()
    plot_tree(tree, ax=ax, filled=True, feature_names=ml_cols, class_names=['Inactive', 'Active'], rounded=True, rotate=True, fontsize=Z-4)
    plt.show()
    plt.close()

if __name__ == '__main__':
    print('This statement will be executed only if this script is called directly')
