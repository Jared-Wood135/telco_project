# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
1. Orientation
2. Imports
3. get_telco
'''

# =======================================================================================================
# Table of Contents END
# Table of Contents TO Orientation
# Orientation START
# =======================================================================================================

'''
The purpose of this file is to ensure that users have the 'telco' dataset is acquired properly
given that the user has the proper 'env.py' file to utilize
'''

# =======================================================================================================
# Orientation END
# Orientation TO Imports
# Imports START
# =======================================================================================================

import pandas as pd
import env
import os

# =======================================================================================================
# Imports END
# Imports TO get_telco
# get_telco START
# =======================================================================================================

def get_telco_data():
    '''
    Creates 'telco.csv' if it does not exist then reads the csv as a pandas dataframe
    '''
    if os.path.exists('telco.csv'):
        return pd.read_csv('telco.csv', index_col=0)
    else:
        telco_df = pd.read_sql(
            '''
            SELECT
                *
            FROM
                customers
                LEFT JOIN (
                        SELECT 
                            customer_id, 
                            DATE(churn_month) AS churn_month 
                        FROM 
                            customer_churn
                            ) AS A USING(customer_id)
                LEFT JOIN (
                        SELECT 
                            customer_id, 
                            DATE(signup_date) AS signup_date 
                        FROM 
                            customer_signups
                            ) AS B USING(customer_id)
                LEFT JOIN contract_types USING(contract_type_id)
                LEFT JOIN internet_service_types USING(internet_service_type_id)
                LEFT JOIN payment_types USING(payment_type_id)
            ''', env.get_db_url('telco_churn')
        )
        telco_df.to_csv('telco.csv')
        return pd.read_csv('telco.csv', index_col=0)
    
# =======================================================================================================
# get_telco END
# =======================================================================================================