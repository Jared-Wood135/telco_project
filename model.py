# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
1. Orientation
2. Imports
3. decisiontree_model_iterator
4. randomforest_model_iterator
5. knn_model_iterator
'''

# =======================================================================================================
# Table of Contents END
# Table of Contents TO Orientation
# Orientation START
# =======================================================================================================

'''
The purpose of this file is to help with modeling and speed up the process of creating models,
visualizations, and possible best models to utilize
'''

# =======================================================================================================
# Orientation END
# Orientation TO Imports
# Imports START
# =======================================================================================================

# Basic sheiza
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Stat/Exploration
from scipy import stats

# Modeling
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# .py files
import acquire
import prepare
import explore

# =======================================================================================================
# Imports END
# Imports TO decisiontree_model_iterator
# decisiontree_model_iterator START
# =======================================================================================================

def decisiontree_model_iterator(df, x_col, y_col, stratify, mindepthrange, maxdepthrange):
    # VVV Variables VVV
    train, validate, test = prepare.split(df, stratify)
    x_train = train[x_col]
    y_train = train[y_col]
    x_validate = validate[x_col]
    y_validate = validate[y_col]
    x_test = test[x_col]
    y_test = test[y_col]
    modelnum = 0
    modelsdict = {}
    # VVV Create, Fit, Predict Models VVV
    for i in range(int(mindepthrange), int(maxdepthrange)):
        modelnum += 1
        clf = DecisionTreeClassifier(max_depth=i)
        clf.fit(x_train, y_train)
        clf.predict(x_train)
        modelsdict['model'] = f'clf{modelnum}'
        modelsdict['train_score'] = round(clf.score(x_train, y_train), 5)
        modelsdict['validate_score'] = round(clf.score(x_validate, y_validate), 5)
        modelsdict['diff'] = round(abs((clf.score(x_train, y_train)) - (clf.score(x_validate, y_validate))), 5)
    return modelsdict

# =======================================================================================================
# decisiontree_model_iterator END
# decisiontree_model_iterator TO randomforest_model_iterator
# randomforest_model_iterator START
# =======================================================================================================

def randomforest_model_iterator():
    print('yeet')

# =======================================================================================================
# randomforest_model_iterator END
# randomforest_model_iterator TO knn_model_iterator
# knn_model_iterator START
# =======================================================================================================

def knn_model_iterator():
    print('yeet')

# =======================================================================================================
# knn_model_iterator END
# =======================================================================================================