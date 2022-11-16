import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
np.random.seed(7)

# My env module
import src.env as env


# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")



####################################
###### SQL Query for Server ########
####################################

def simple_sql_zillow_2017():
    '''
    This function passes a SQL query for specified columns, converts that into a pandas dataframe and then
    returns that dataframe
    '''

    sql_query = '''
    SELECT taxvaluedollarcnt, bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, fips
    FROM properties_2017 AS prop

    JOIN predictions_2017 AS pred ON prop.parcelid = pred.parcelid
        AND pred.transactiondate >= '2017-01-01'

    WHERE prop.bedroomcnt > 0
        AND prop.bathroomcnt >0
        AND prop.propertylandusetypeid = '261'
    ''' 
    
    # reads the returned data tables into a dataframe
    df = pd.read_sql(sql_query, env.codeup_db('zillow'))

    
    # Cache data
    df.to_csv('data/simple_zillow_2017.csv')
    
    return df


####################################################
############       rename columns       ############
####################################################

def simple_rename_columns(df):
    df = df.rename(columns={'bedroomcnt':'bedrooms', 
                            'bathroomcnt':'baths', 
                            'calculatedfinishedsquarefeet':'sq_feet', 
                            #'yearbuilt':'year_built',
                            'taxvaluedollarcnt':'tax_value'})
    return df



#######################################################
############       handling outliers       ############
#######################################################

def simple_handle_outliers(df):
    """Manually handle outliers that do not represent properties likely for 99% of buyers and zillow visitors"""
    df = df[df.bedrooms <= 6]
    df = df[df.baths <= 6]
    df = df[df.tax_value < 2_000_000]
    df = df[df.sq_feet < 10000]

    return df



####################################################
############       handling naans       ############
####################################################

def simple_deal_with_nulls(df):

    # fills whitespace will Naans
    df = df.replace(r'^\s*s', np.NaN, regex=True)

    # the columns which we want to drop naan values from
    #naan_drop_columns = ['sq_feet', 'tax_value', 'year_built', 'tax_amount']    
    naan_drop_columns = ['tax_value', 'sq_feet']    

    # drop naans based on the columns identified above
    df = df.dropna(subset = naan_drop_columns)

    return df


####################################################
#######         cast columns as int          #######
####################################################

def simple_columns_to_int(df):

    # renames columns to be more intelligable and able to be referenced
    df['bedrooms'] = df['bedrooms'].astype(int) 
    df['baths'] = df['baths'].astype(int)
    df['fips'] = df['fips'].astype(int)
    df['tax_value'] = df['tax_value'].astype(int)
    df['sq_feet'] = df['sq_feet'].astype(int)
    #df['year_built'] = df['year_built'].astype(int)
    #df['tax_amount'] = df['tax_amount'].astype(int)

    
    # way to cast all column elements as integers
    #df[(list(df.columns))].astype(int) 
    
    
    return df


#######################################################
#######     calls simple cleaning functions     #######
#######################################################

def simple_cleaning(df):
    # runs functions defined earlier in program which clean up dataframe    
    df = simple_rename_columns(df)
    df = simple_deal_with_nulls(df)
    df = simple_columns_to_int(df)    
    df = simple_handle_outliers(df)

    return df 


#############################################################
###### Cleans zillow dataframe using cleaning modules #######
#############################################################

def zillow_2017(simple = True, small = False):
    '''
    This is a very large dataset and the values can get very wide so we will handle null values by dropping them.
    The process will be to first turn whitespace into null values and then drop rows with null values from columns 
    which we find to be important.  
    '''

    if simple == True:
        # checks to see if wrangled zillow data exists already
        # if it does, then fills df with the stored data
        # retains "small" variable option
        if os.path.isfile('data/simple_wrangled_zillow_2017.csv'):

            df = pd.read_csv('data/simple_wrangled_zillow_2017.csv')

            if small == True:
                df = df.sample(frac=0.5)

            return df

        else:
            df = simple_sql_zillow_2017()

        # calls function to clean dirty data
        df = simple_cleaning(df)
    
        # if a smaller sized sample of the data is sought to be used to conserve computational resources,
        # the small variable can be modified to cut the dataframe down to half its original size
        if small == True:
            df = df.sample(frac=0.5)

        # Cache data so future runs of this program go by more quickly
        df.to_csv('data/simple_wrangled_zillow_2017.csv')


        return df 




###########################################################################
#######      Functions for Splitting Data for Machine Learning      #######
###########################################################################

##############              First Split              ##############

def split(df):
    '''
    function to split dataframe into portions for training a model, validating the model, 
    with the goal of ultimately testing a good model
    ''' 
    
    # splits data into two groups, holding the test variable to the side
    train_validate, test = train_test_split(df, test_size = 0.2)
    
    # splits train_validate into two groups, train and validate
    train, validate = train_test_split(train_validate, test_size = 0.3)
    
    # returns train, validate, and test variables
    return train, validate, test



##############              Second Split              ##############

def x_y(df, target):
    '''
    This function depends on the split function, being handed a dataframe
    and the target variable we are seeking to understand through prediction
    '''

    # calls split function to produce required variables
    train, validate, test = split(df)

    x_train = train.drop(columns=[target])
    y_train = train[target]
    
    x_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    x_test = test.drop(columns=[target])
    y_test = test[target]
    
    return x_train, y_train, x_validate, y_validate, x_test, y_test