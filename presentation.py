# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
1. Orientation
2. Imports
3. explore
4. top3models
'''

# =======================================================================================================
# Table of Contents END
# Table of Contents TO Orientation
# Orientation START
# =======================================================================================================

'''
The purpose of this file is to prepare specific function calls to visualizations, statistics, and models
for easier presentation purposes
'''

# =======================================================================================================
# Orientation END
# Orientation TO Imports
# Imports START
# =======================================================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from scipy import stats
import prepare

# =======================================================================================================
# Imports END
# Imports TO explore
# explore START
# =======================================================================================================

def explore():
    telco = prepare.prep_telco()
    top4 = [
        'contract_type',
        'sign_year',
        'tenure',
        'internet_service_type'
    ]
    for x in top4:
        sns.histplot(data=telco, x=x, hue='churn', multiple='dodge')
        plt.title(f'{x} vs. churn')
        plt.show()
        observed = pd.crosstab(telco[x], telco.churn)
        alpha = 0.05
        p_val = stats.chi2_contingency(observed)[1]
        if alpha > p_val:
            print(f'\033[32m{x}\033[0m p_val: {p_val}\n')
        else:
            print(f'\033[31m{x}\033[0m does not have an impact: {p_val}\n')

# =======================================================================================================
# explore END
# explore TO top3models
# top3models START
# =======================================================================================================

def top3models():
    telco = prepare.prep_telco()
    train_val, test = train_test_split(telco, train_size=0.8, random_state=1349, stratify=telco['churn'])
    train, val = train_test_split(train_val, train_size=0.7, random_state=1349, stratify=train_val['churn'])
    scores = {
    'model' : ['actual', 'baseline'],
    'test' : [100, 73.5],
    'TP("No")' : [100, 73.5],
    'TN("Yes")' : [100, 0.0],
    }
    keylist = [
    'online_security_No',
    'online_backup_No',
    'device_protection_No',
    'tech_support_No',
    'contract_type_Month-to-month',
    'internet_service_type_Fiber_optic',
    'payment_type_Electronic_check',
    'sign_year',
    'tenure',
    'value_per_total_services'
    ]
    x_train = train[keylist]
    y_train = train['churn']
    x_val = val[keylist]
    y_val = val['churn']
    x_test = test[keylist]
    y_test = test['churn']  
    rfc = RFC(max_depth=7, random_state=100)
    rfc.fit(x_train, y_train)
    model = rfc.predict(x_test)
    testscore = (round(rfc.score(x_test, y_test), 2) * 100)
    matrix = confusion_matrix(y_test, model, labels=('Yes', 'No'))
    TN = (round(matrix[0, 0] / (matrix[0, 0] + matrix[1, 0]), 3) * 100)
    TP = (round(matrix[1, 1] / (matrix[1, 1] + matrix[0, 1]), 3) * 100)
    scores['model'].append('RFC')
    scores['test'].append(testscore)
    scores['TP("No")'].append(TP)
    scores['TN("Yes")'].append(TN)
    lr = LR(random_state=100)
    lr.fit(x_train, y_train)
    model = lr.predict(x_test)
    testscore = (round(lr.score(x_test, y_test), 3) * 100)
    matrix = confusion_matrix(y_test, model, labels=('Yes', 'No'))
    TN = (round(matrix[0, 0] / (matrix[0, 0] + matrix[1, 0]), 3) * 100)
    TP = (round(matrix[1, 1] / (matrix[1, 1] + matrix[0, 1]), 3) * 100)
    scores['model'].append('LR')
    scores['test'].append(testscore)
    scores['TP("No")'].append(TP)
    scores['TN("Yes")'].append(TN)
    dtc = DTC(max_depth=3, random_state=100)
    dtc.fit(x_train, y_train)
    model = dtc.predict(x_test)
    testscore = (round(dtc.score(x_test, y_test), 3) * 100)
    matrix = confusion_matrix(y_test, model, labels=('Yes', 'No'))
    TN = (round(matrix[0, 0] / (matrix[0, 0] + matrix[1, 0]), 3) * 100)
    TP = (round(matrix[1, 1] / (matrix[1, 1] + matrix[0, 1]), 3) * 100)
    scores['model'].append('DTC')
    scores['test'].append(testscore)
    scores['TP("No")'].append(TP)
    scores['TN("Yes")'].append(TN)
    return pd.DataFrame.from_dict(scores)