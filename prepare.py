# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
1. Orientation
2. Imports
3. prep_telco
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
    return telco_db

# =======================================================================================================
# prep_telco END
# =======================================================================================================