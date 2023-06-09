# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
1. Orientation
2. Imports
3. prep_telco
4. split
'''

# =======================================================================================================
# Table of Contents END
# Table of Contents TO Orientation
# Orientation START
# =======================================================================================================

'''
The purpose of this file is to prepare the 'telco' dataset in order to ensure consistency with
the initial data for anyone attempting to replicate the process...
'''

# =======================================================================================================
# Orientation END
# Orientation TO Imports
# Imports START
# =======================================================================================================

import pandas as pd
import acquire
from sklearn.model_selection import train_test_split
import os

# =======================================================================================================
# Imports END
# Imports TO prep_telco
# prep_telco START
# =======================================================================================================

def prep_telco():
    '''
    Takes the 'telco.csv' dataframe from 'acquire.py' and prepares the dataframe for use with
    consistent data structuring
    '''
    telco_db = acquire.get_telco_data()
    telco_db = telco_db.drop(columns=['customer_id', 'contract_type_id', 'payment_type_id', 'internet_service_type_id'])
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
    dummies = pd.get_dummies(telco_db.select_dtypes(include='object'))
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
    return telco_db

# =======================================================================================================
# prep_telco END
# prep_telco TO prep_split
# prep_split START
# =======================================================================================================

def split(df, stratify):
    '''
    Takes a dataframe and splits the data into a train, validate and test datasets
    '''
    train_val, test = train_test_split(df, train_size=0.8, random_state=1349, stratify=df[stratify])
    train, validate = train_test_split(train_val, train_size=0.7, random_state=1349, stratify=train_val[stratify])
    print(f"train.shape:{train.shape}\nvalidate.shape:{validate.shape}\ntest.shape:{test.shape}")
    return train, validate, test

# =======================================================================================================
# prep_split END
# =======================================================================================================