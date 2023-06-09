# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
1. Orientation
2. Imports
3. explore
4. top3models
5. pred_csv
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

import acquire
import os
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

def explore1():
    '''
    Return 1st key visualization
    '''
    telco = prepare.prep_telco()
    sns.boxplot(data=telco, x='tenure', y='churn')
    plt.title('Customer Tenure vs. Customer Churn')
    plt.show()
    observed = pd.crosstab(telco.tenure, telco.churn)
    p_val = stats.chi2_contingency(observed)[1]
    print(f'\033[32mTenure p_val\033[0m: {p_val:.3}\n')
    
def explore2():
    '''
    Return 2nd key visualization
    '''
    telco = prepare.prep_telco()
    sns.histplot(data=telco, x='sign_year', hue='churn', multiple='dodge', kde=True)
    plt.title('Customer Signup Year vs. Customer Churn')
    plt.show()
    observed = pd.crosstab(telco.sign_year, telco.churn)
    p_val = stats.chi2_contingency(observed)[1]
    print(f'\033[32mSignup Year p_val\033[0m: {p_val:.3}\n')

def explore3():
    '''
    Return 2nd key visualization
    '''
    telco = prepare.prep_telco()
    sns.barplot(data=telco, x='sign_year', y='monthly_charges')
    plt.title('Customer Monthly Charges(USD) vs. Customer Signup Year')
    plt.show()

def explore4():
    '''
    Return 2nd key visualization
    '''
    telco = prepare.prep_telco()
    sns.histplot(data=telco, x='total_extra_services', hue='churn', multiple='dodge', kde=True)
    plt.title('Customer Internet add-ons vs. Customer Churn')
    plt.show()
    observed = pd.crosstab(telco.total_extra_services, telco.churn)
    p_val = stats.chi2_contingency(observed)[1]
    print(f'\033[32mTotal Extra Services p_val\033[0m: {p_val:.3}\n')

def explore5():
    '''
    Return 2nd key visualization
    '''
    telco = prepare.prep_telco()
    sns.histplot(data=telco, x='contract_type', hue='churn', multiple='dodge')
    plt.title('Customer Contract Type vs. Customer Churn')
    plt.show()
    observed = pd.crosstab(telco.contract_type, telco.churn)
    p_val = stats.chi2_contingency(observed)[1]
    print(f'\033[32mContract Type p_val\033[0m: {p_val:.3}\n')

def explore6():
    '''
    Return 2nd key visualization
    '''
    telco = prepare.prep_telco()
    sns.histplot(data=telco, x='sign_year', hue='contract_type', multiple='dodge', kde=True)
    plt.title('Customer Signup Year vs. Customer Contract Type')
    plt.show()

# =======================================================================================================
# explore END
# explore TO top3models
# top3models START
# =======================================================================================================

def top3models():
    '''
    Returns the top 3 models and their scores...
    '''
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

# =======================================================================================================
# top3models END
# top3models TO pred_csv
# pred_csv START
# =======================================================================================================

def pred_csv():
    '''
    Creates a predictions.csv with customer_id, actual churn, and predictions
    '''
    if os.path.exists('predictions.csv'):
        return pd.read_csv('predictions.csv', index_col=0)
    else:
        telco_db = acquire.get_telco_data()
        telco_db = telco_db.drop(columns=['contract_type_id', 'payment_type_id', 'internet_service_type_id'])
        clean_charges = []
        floatnumbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '.']
        for charge in telco_db.total_charges:
            valuestr = ''
            for char in charge:
                if char in floatnumbers:
                    valuestr += char
            if valuestr:
                clean_charges.append(float(valuestr))
            else:
                clean_charges.append(None)
        telco_db.total_charges = clean_charges
        telco_db.total_charges = telco_db.total_charges.fillna(0)
        telco_db.churn_month = pd.to_datetime(telco_db.churn_month)
        telco_db.signup_date = pd.to_datetime(telco_db.signup_date)
        telco_db['sign_year'] = pd.DatetimeIndex(telco_db['signup_date']).year.astype('object')
        telco_db['sign_month'] = pd.DatetimeIndex(telco_db['signup_date']).month.astype('object')
        telco_db['sign_day'] = pd.DatetimeIndex(telco_db['signup_date']).day.astype('object')
        telco_db['sign_dayofweek'] = pd.DatetimeIndex(telco_db['signup_date']).dayofweek.astype('object')
        dummies = pd.get_dummies(telco_db.drop(columns='customer_id').select_dtypes(include='object'))
        telco_db = pd.concat([telco_db, dummies], axis=1)
        telco_db['total_services'] = (telco_db.phone_service_Yes 
                            + telco_db.multiple_lines_Yes
                            + telco_db.online_security_Yes 
                            + telco_db.online_backup_Yes
                            + telco_db.device_protection_Yes
                            + telco_db.tech_support_Yes
                            + telco_db.streaming_tv_Yes
                            + telco_db.streaming_movies_Yes 
                            + telco_db.internet_service_type_DSL 
                            + telco_db['internet_service_type_Fiber optic'])
        telco_db.total_services = telco_db.total_services.astype(int)
        telco_db['total_extra_services'] = (telco_db.online_security_Yes 
                                    + telco_db.online_backup_Yes
                                    + telco_db.device_protection_Yes
                                    + telco_db.tech_support_Yes
                                    + telco_db.streaming_tv_Yes
                                    + telco_db.streaming_movies_Yes)
        telco_db.total_extra_services = telco_db.total_extra_services.astype(int)
        telco_db['value_per_total_services'] = telco_db.monthly_charges / telco_db.total_services
        telco_db['value_per_total_extra_services'] = telco_db.monthly_charges / telco_db.total_extra_services
        telco_db.columns = telco_db.columns.str.replace(' ', '_')
        telco = telco_db
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
        train_val, test = train_test_split(telco, train_size=0.8, random_state=1349, stratify=telco['churn'])
        train, val = train_test_split(train_val, train_size=0.7, random_state=1349, stratify=train_val['churn'])
        x_train = train[keylist]
        y_train = train['churn']
        x_test = test[keylist]
        y_test = test['churn']  
        rfc = RFC(max_depth=7, random_state=100)
        rfc.fit(x_train, y_train)
        model = rfc.predict(x_test)
        probab = rfc.predict_proba(x_test)
        yesprob = probab[:, 1]
        predictions = test['customer_id']
        pred_df = pd.DataFrame(predictions)
        pred_df['probability_yes'] = yesprob
        pred_df['prediction'] = model
        pred_df.to_csv('predictions.csv')
        return pd.read_csv('predictions.csv', index_col=0)

# =======================================================================================================
# pred_csv END
# =======================================================================================================