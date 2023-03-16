# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
1. Orientation
2. Imports
3. quantitative
4. categorical
5. val_vs_cat
6. mannwhitneyu
'''

# =======================================================================================================
# Table of Contents END
# Table of Contents TO Orientation
# Orientation START
# =======================================================================================================

'''
The purpose of this file is to make exploring data from a dataframe easier and faster to do
'''

# =======================================================================================================
# Orientation END
# Orientation TO Imports
# Imports START
# =======================================================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools
from scipy import stats
import acquire
import prepare

# =======================================================================================================
# Imports END
# Imports TO quantitative
# quantitative START
# =======================================================================================================

def quantitative(df):
    '''
    Takes in a dataframe and gets all of the float and int dtype columns then 
    returns a histogram, boxplot, and a .describe for each column
    '''
    quantitative_col = df.select_dtypes(include=['float', 'int'])
    for col in quantitative_col:
        sns.histplot(quantitative_col[col])
        plt.title(f'Distribution of {col}')
        plt.show()
        sns.boxplot(quantitative_col[col])
        plt.title(f'Distribution of {col}')
        plt.show()
        print(quantitative_col[col].describe().to_markdown())
        print('\n=======================================================\n')

# =======================================================================================================
# quantitative END
# quantitative TO categorical
# categorical START
# =======================================================================================================

def categorical(df):
    '''
    Takes in a dataframe and gets all of the object dtype columns then 
    returns a histogram and a total count and percentage table for each unique value
    '''
    categorical_col = df.select_dtypes(include='object')
    for col in categorical_col:
        sns.histplot(categorical_col, x=col, hue=col, legend=False)
        plt.title(f'Count of values in {col}')
        plt.show()
        colval = categorical_col[col].value_counts()
        colpct = categorical_col[col].value_counts(normalize=True)
        table = pd.concat([colval, colpct], axis=1)
        table.columns = ['Count', 'Percentage']
        print(table)
        print('\n=======================================================\n')

# =======================================================================================================
# categorical END
# categorical TO val_vs_cat
# val_vs_cat START
# =======================================================================================================

def cat_vs_val(df):
    '''
    Takes in a dataframe and separates all of the columns into category or values then
    returns a bargraph of each unique combination of category and value columns
    '''
    val_col = []
    cat_col = []
    for col in df:
        if df[col].dtype == 'O':
            cat_col.append(col)
        elif (df[col].dtype == 'float64') | (df[col].dtype == 'int'):
            val_col.append(col)
    combo_col = list(itertools.product(cat_col, val_col))
    for x, y in combo_col:
        sns.barplot(x=df[x], y=df[y])
        plt.title(f'Relationship of {x} and {y}')
        plt.axhline(y=df[y].mean(), color='r', label=f'Mean: {round(df[y].mean(), 2)}')
        plt.legend()
        plt.show()
        print(df.groupby(x)[y].describe().T.to_markdown())
        print('\n=======================================================\n')


# =======================================================================================================
# val_vs_cat END
# val_vs_cat TO mannwhitneyu
# mannwhitneyu START
# =======================================================================================================

def mannwhitneyu(df):
    '''
    Takes in a dataframe and user's desired columns and returns manwhitneyu results for each
    unique combination of the inputted column.
    '''
    cat_col = input("What column do you want unique combinations of?\n")
    combos = itertools.combinations(df[cat_col].unique(), 2)
    val_col = []
    for col in df:
        if (df[col].dtype == 'float64') | (df[col].dtype == 'int'):
            val_col.append(col)
    for x in combos:
        for y in val_col:
            print(f'\033[32m{x[0]}\033[0m and \033[32m{x[1]}\033[0m by \033[32m{y}\033[0m relationship:')
            stat, pval = stats.mannwhitneyu(df[df[cat_col] == x[0]][y], df[df[cat_col] == x[1]][y])
            print(f'\033[32mStat:\033[0m {stat}\n\033[32mP-value:\033[0m {pval}\n')



def swarms(df):
    cat_col = []
    val_col = []
    for col in df:
        if df[col].dtype == 'object':
            cat_col.append(col)
        elif ((df[col].dtype == ('float64')) | (df[col].dtype == ('int'))):
            val_col.append(col)
    pairs = list(itertools.product(cat_col, val_col))
    for x, y in pairs:
        print(f'Swarmplot of \033[32m{x}\033[0m and \033[32m{y}\033[0m Relationship')
        sns.swarmplot(df[x], df[y])
        plt.show()