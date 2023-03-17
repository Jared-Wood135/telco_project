# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
1. Orientation
2. Imports
3. specificity
4. accuracy
5. precision
6. recall
7. sensitivity_true_positive_rate
8. negative_predictive_value
9. f1_score
10. train_val_scores
'''

# =======================================================================================================
# Table of Contents END
# Table of Contents TO Orientation
# Orientation START
# =======================================================================================================

'''
The purpose of this file is to help with model evaluations and output their respective percents...

Friendly reminder...
sklearn.metrics.confusion_matrix returns a 2x2 array which then means:
    - True Positive (TP) = [1, 1]
    - False Positive (FP) = [0, 1]
    - True Negative (TN) = [0, 0]
    - False Negative (FN) = [1, 0]
    - Labels Define columns via [Negative Actual, Positive Actual]
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# .py files
import acquire
import prepare
import explore

# =======================================================================================================
# Imports END
# Imports TO specificity
# specificity START
# =======================================================================================================

def specificity(df, actual_col, Falselabel, Truelabel):
    '''
    Iterates through all columns with the specificity model metric...
    Specificity measures how well a model predicts negative outcomes...
    Specificity = (TN / (FP + TN))
    '''
    ratios_dict = {}
    for col in df:
        matrix = confusion_matrix(df[actual_col], df[col], labels=(Falselabel, Truelabel))
        result = (matrix[0, 0] / (matrix[0, 1] + matrix[0, 0]))
        print(f'\033[32m{col}:\033[0m {result:.2%}\n')
        ratios_dict[col] = result
    del ratios_dict[actual_col]
    max_col = max(ratios_dict, key=ratios_dict.get)
    max_ratio = ratios_dict[max_col]
    min_col = min(ratios_dict, key=ratios_dict.get)
    min_ratio = ratios_dict[min_col]
    print(f'\033[31mHIGHEST VALUE =\033[0m \033[32m{max_col}\033[0m: {max_ratio:.2%}\n\033[31mLOWEST VALUE =\033[0m \033[32m{min_col}\033[0m: {min_ratio:.2%}')

# =======================================================================================================
# specificity END
# specificity TO accuracy
# accuracy START
# =======================================================================================================

def accuracy(df, actual_col):
    '''
    Iterates through all columns with the accuracy model metric...
    Accuracy measures how many correct predictions over total possible predictions...
    Accuracy = ((TP + TN) / (TP + FP + FN + TN))
    '''
    ratios_dict = {}
    for col in df:
        ratio = (df[col] == df[actual_col]).mean()
        print(f'\033[32m{col}:\033[0m {ratio:.2%}\n')
        ratios_dict[col] = ratio
    del ratios_dict[actual_col]
    max_col = max(ratios_dict, key=ratios_dict.get)
    max_ratio = ratios_dict[max_col]
    min_col = min(ratios_dict, key=ratios_dict.get)
    min_ratio = ratios_dict[min_col]
    print(f'\033[31mHIGHEST VALUE =\033[0m \033[32m{max_col}\033[0m: {max_ratio:.2%}\n\033[31mLOWEST VALUE =\033[0m \033[32m{min_col}\033[0m: {min_ratio:.2%}')

# =======================================================================================================
# accuracy END
# accuracy TO precision
# precision START
# =======================================================================================================

def precision(df, actual_col, Falselabel, Truelabel):
    '''
    Iterates through all columns with the precision model metric...
    Precision measures how many of the positive predictions were correct...
    Precision = (TP / (TP + FP))
    '''
    ratios_dict = {}
    for col in df:
        matrix = confusion_matrix(df[actual_col], df[col], labels=(Falselabel, Truelabel))
        ratio = (matrix[1, 1] / (matrix[1, 1] + matrix[0, 1]))
        print(f'\033[32m{col}:\033[0m {ratio:.2%}\n')
        ratios_dict[col] = ratio
    del ratios_dict[actual_col]
    max_col = max(ratios_dict, key=ratios_dict.get)
    max_ratio = ratios_dict[max_col]
    min_col = min(ratios_dict, key=ratios_dict.get)
    min_ratio = ratios_dict[min_col]
    print(f'\033[31mHIGHEST VALUE =\033[0m \033[32m{max_col}\033[0m: {max_ratio:.2%}\n\033[31mLOWEST VALUE =\033[0m \033[32m{min_col}\033[0m: {min_ratio:.2%}')

# =======================================================================================================
# precision END
# precision TO recall
# recall START
# =======================================================================================================

def recall(df, actual_col, Falselabel, Truelabel):
    '''
    Iterates through all columns with the recall model metric...
    Recall measures how the model handled all positive outcomes...
    Recall = (TP / (TP + FN))
    '''
    ratios_dict = {}
    for col in df:
        matrix = confusion_matrix(df[actual_col], df[col], labels=(Falselabel, Truelabel))
        ratio = (matrix[1, 1] / (matrix[1, 1] + matrix[1, 0]))
        print(f'\033[32m{col}:\033[0m {ratio:.2%}\n')
        ratios_dict[col] = ratio
    del ratios_dict[actual_col]
    max_col = max(ratios_dict, key=ratios_dict.get)
    max_ratio = ratios_dict[max_col]
    min_col = min(ratios_dict, key=ratios_dict.get)
    min_ratio = ratios_dict[min_col]
    print(f'\033[31mHIGHEST VALUE =\033[0m \033[32m{max_col}\033[0m: {max_ratio:.2%}\n\033[31mLOWEST VALUE =\033[0m \033[32m{min_col}\033[0m: {min_ratio:.2%}')

# =======================================================================================================
# recall END
# recall TO sensitivity_true_positive_rate
# sensitivity_true_positive_rate START
# =======================================================================================================

def sensitivity_true_positive_rate(df, actual_col, Falselabel, Truelabel):
    '''
    Iterates through all columns with the sensitivity true positive rate model metric...
    Sensitivity true positive rate measures the proportion of positives correctly identified...
    Sensitivity true positive rate = (TP / (TP + FN))
    '''
    ratios_dict = {}
    for col in df:
        matrix = confusion_matrix(df[actual_col], df[col], labels=(Falselabel, Truelabel))
        ratio = (matrix[1, 1] / (matrix[1, 1] + matrix[1, 0]))
        print(f'\033[32m{col}:\033[0m {ratio:.2%}\n')
        ratios_dict[col] = ratio
    del ratios_dict[actual_col]
    max_col = max(ratios_dict, key=ratios_dict.get)
    max_ratio = ratios_dict[max_col]
    min_col = min(ratios_dict, key=ratios_dict.get)
    min_ratio = ratios_dict[min_col]
    print(f'\033[31mHIGHEST VALUE =\033[0m \033[32m{max_col}\033[0m: {max_ratio:.2%}\n\033[31mLOWEST VALUE =\033[0m \033[32m{min_col}\033[0m: {min_ratio:.2%}')

# =======================================================================================================
# sensitivity_true_positive_rate END
# sensitivity_true_positive_rate TO negative_predictive_value
# negative_predictive_value START
# =======================================================================================================

def negative_predictive_value(df, actual_col, Falselabel, Truelabel):
    '''
    Iterates through all columns with the negative predictive value model metric...
    Negative predictive value measures the probability that a predicted negative is a true negative...
    Negative predictive value = (TN / (TN + FN))
    '''
    ratios_dict = {}
    for col in df:
        matrix = confusion_matrix(df[actual_col], df[col], labels=(Falselabel, Truelabel))
        ratio = (matrix[0, 0] / (matrix[0, 0] + matrix[1, 0]))
        print(f'\033[32m{col}:\033[0m {ratio:.2%}\n')
        ratios_dict[col] = ratio
    del ratios_dict[actual_col]
    max_col = max(ratios_dict, key=ratios_dict.get)
    max_ratio = ratios_dict[max_col]
    min_col = min(ratios_dict, key=ratios_dict.get)
    min_ratio = ratios_dict[min_col]
    print(f'\033[31mHIGHEST VALUE =\033[0m \033[32m{max_col}\033[0m: {max_ratio:.2%}\n\033[31mLOWEST VALUE =\033[0m \033[32m{min_col}\033[0m: {min_ratio:.2%}')

# =======================================================================================================
# negative_predictive_value END
# negative_predictive_value TO f1_score
# f1_score START
# =======================================================================================================

def f1_score(df, actual_col, Falselabel, Truelabel):
    '''
    Iterates through all columns with the f1 score model metric...
    F1 score measures a model's accuracy on a dataset...
    F1 score = 2 * ((Precision * Recall) / (Precision + Recall))
    '''
    ratios_dict = {}
    for col in df:
        matrix = confusion_matrix(df[actual_col], df[col], labels=(Falselabel, Truelabel))
        precision = (matrix[1, 1] / (matrix[1, 1] + matrix[0, 1]))
        recall = (matrix[1, 1] / (matrix[1, 1] + matrix[1, 0]))
        ratio = (2 * ((precision * recall) / (precision + recall)))
        print(f'\033[32m{col}:\033[0m {ratio:.2%}\n')
        ratios_dict[col] = ratio
    del ratios_dict[actual_col]
    max_col = max(ratios_dict, key=ratios_dict.get)
    max_ratio = ratios_dict[max_col]
    min_col = min(ratios_dict, key=ratios_dict.get)
    min_ratio = ratios_dict[min_col]
    print(f'\033[31mHIGHEST VALUE =\033[0m \033[32m{max_col}\033[0m: {max_ratio:.2%}\n\033[31mLOWEST VALUE =\033[0m \033[32m{min_col}\033[0m: {min_ratio:.2%}')

# =======================================================================================================
# f1_score END
# f1_score TO train_val_scores
# train_val_scores START
# =======================================================================================================

def train_val_scores(train_df, validate_df, x_cols, y_cols, models_list):
    '''
    Returns train and validate scores for all of the models in the models_list as well as the top and lowest performers...
    '''
    train_dict = {}
    validate_dict = {}
    performance_dict = {}
    modelnum = 0
    for x in models_list:
        modelnum += 1
        train_score = round(x.score(train_df[x_cols], train_df[y_cols]), 2)
        validate_score = round(x.score(validate_df[x_cols], validate_df[y_cols]), 2)
        train_dict[modelnum] = train_score
        validate_dict[modelnum] = validate_score
        diff = abs(train_score - validate_score)
        performance_dict[modelnum] = diff
        print(f'\033[32mmodel{modelnum}\033[0m Train Score: {train_score:.2%}')
        print(f'\033[32mmodel{modelnum}\033[0m Validate Score: {validate_score:.2%}\n')
    train_max_model = max(train_dict, key=train_dict.get)
    train_max_pct = train_dict[train_max_model]
    train_min_model = min(train_dict, key=train_dict.get)
    train_min_pct = train_dict[train_min_model]
    validate_max_model = max(validate_dict, key=validate_dict.get)
    validate_max_pct = validate_dict[validate_max_model]
    validate_min_model = min(validate_dict, key=validate_dict.get)
    validate_min_pct = validate_dict[validate_min_model]
    performance_lowest_model = max(performance_dict, key=performance_dict.get)
    performance_lowest_pct = performance_dict[performance_lowest_model]
    performance_highest_model = min(performance_dict, key=performance_dict.get)
    performance_highest_pct = performance_dict[performance_highest_model]
    print(f'\033[31mHIGHEST VALUE (TRAIN)\033[0m = \033[32mmodel{train_max_model}\033[0m: {train_max_pct:.2%}')
    print(f'\033[31mLOWEST VALUE (TRAIN)\033[0m = \033[32mmodel{train_min_model}\033[0m: {train_min_pct:.2%}')
    print(f'\033[31mHIGHEST VALUE (VALIDATE)\033[0m = \033[32mmodel{validate_max_model}\033[0m: {validate_max_pct:.2%}')
    print(f'\033[31mLOWEST VALUE (VALIDATE)\033[0m = \033[32mmodel{validate_min_model}\033[0m: {validate_min_pct:.2%}')
    print(f'\033[31mHIGHEST DIFF\033[0m = \033[32mmodel{performance_lowest_model}\033[0m: {performance_lowest_pct:.5%}')
    print(f'\033[31mLOWEST DIFF\033[0m = \033[32mmodel{performance_highest_model}\033[0m: {performance_highest_pct:.5%}')

# =======================================================================================================
# train_val_scores END
# =======================================================================================================