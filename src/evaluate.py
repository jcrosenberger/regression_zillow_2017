'''
Create a file named evaluate.py that contains the following functions.

- plot_residuals(y, yhat): creates a residual plot
- regression_errors(y, yhat): returns the following values:
    sum of squared errors (SSE)
    explained sum of squares (ESS)
    total sum of squares (TSS)
    mean squared error (MSE)
    root mean squared error (RMSE)
- baseline_mean_errors(y): computes the SSE, MSE, and RMSE for the baseline model
- better_than_baseline(y, yhat): returns true if your model performs better than the baseline, otherwise false
'''

import seaborn as sns
import pandas as pd
import numpy as np
from math import sqrt

#evaluation libraries
from sklearn.metrics import mean_squared_error, r2_score

# linear regressions
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor

# decision tree-based regressions
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor




##############################################
######      Continuous Test DataFrame      ########
##############################################
def pearson_test_df(df, target_var, test_var_list):
    '''default test for continuous to continuous correlation tests. 
    Handles linear relationships well'''
    
    pearson_df = pd.DataFrame(
        {'Potential_Feature':[],
         'Coefficient' :[],
         'P-Value' : [],
         'Significance' : [],
         'Keep' : [],})

    for item in test_var_list:
        r, p_value = pearsonr(df[target_var], df[item])
        if 1 - p_value >= 0.95:
            keeper = 'Yes'
        else:
            keeper = 'No'
        
        pearson_df = pearson_df.append(
        {'Potential_Feature': item,
         'Coefficient' : r,
         'P-Value' : p_value,
         'Significance' : 1-p_value,
         'Keep' : keeper},
        ignore_index = True)
        
    return pearson_df


########################################################
######       Categorical Test DataFrame       ########
########################################################

#######       Chi^2 is an easier test to use       #######

def chi2_categorical_test(df, target_var, test_var_list):
    '''
    The chi2 test is used to determine if a statistically significant relationship 
    exists between two categorical variables
    
    This function takes in a list of variables to test against a singular target variable
    returning a dataframe which should help to determine if the list of variables should
    be accepted or rejected for use in a model to explain the target variable
    '''
    
    chi2_df = pd.DataFrame(columns =[
         'Potential_Feature', 'Chi2_stat', 'P-Value', 'Significance', 'Keep'])
    
    
    for item in test_var_list:
        ctab = pd.crosstab(df[item],df[target_var])
        chi, p_value, degf, expected = chi2_contingency(ctab)
        
        if 1 - p_value >= 0.95:
            keeper = 'Yes'
        else:
            keeper = 'No'
            
        # potential = item, 
        # significance = 1-p_value
        # keep = keeper
        chi2_df.loc[len(chi2_df)] = [item, chi, p_value, 1-p_value, keeper]
        
    return chi2_df.sort_values(by='Keep', ascending = False)



def plot_residuals(y, yhat):
    fig = sns.scatterplot(x = y, y = yhat)



def calc_performance(y, yhat, featureN = 2):
    # explained sum of squares
    ess = ((yhat - y.mean())**2).sum()
    
    # sum of squres errors
    sse = mean_squared_error(y, yhat)*len(y)
    
    # total sum of squares
    tss = ess+sse
    
    # mean sum of squares error
    mse = mean_squared_error(y,yhat)
    
    # rooted sum of squares 
    rmse = sqrt(mse)
    
    # R squared
    r2 = ess/tss
    # A second version of R Squared which may be more accurate
    r2v2 = r2_score(y, yhat)
    
    if featureN > 2:
        # Adjusted R Squared
        adjR2= 1-(1-r2v2)*(len(y)-1)/(len(y)-featureN-1)

        return ess, sse, tss, mse, rmse, r2v2, adjR2    
    
    else:
        return ess, sse, tss, mse, rmse, r2v2



def regression_errors(y, yhat, df=False, features=2):
    '''
    This module does the legwork for evaluating model efficacy. 
    The default argument 'df' will determine whether to pass a data frame with a dictionary 
    when called or to print the evaluation metrics. 
    The AdjR2Feature default argument allows for tuning an Adjusted R^2 value, which is more accurate
    than R^2 for models with multiple features/variables 

    '''
    
    
    if features <= 2:
        ess, sse, tss, mse, rmse, r2v2 = calc_performance(y, yhat)
        if df==False:
            print(f'''Model Performance
            ESS = {round(ess,5)}
            SSE = {round(sse,5)}
            TSS = {round(tss,5)}
            MSE = {round(mse,5)}
            RMSE = {round(rmse,5)}
            R2 = {round(r2v2,10)}''')
        

        else:
            df = pd.DataFrame()
        
            df ={
                'ESS' : round(ess,3),
                'SSE' : round(sse,3),
                'TSS' : round(tss,3),
                'MSE' : round(mse,3),
                'RMSE': round(rmse,3),
                'R2': round(r2v2,3)
                }
                
            return df

    else: 
        ess, sse, tss, mse, rmse, r2v2, adjR2 = calc_performance(y, yhat, features)
        if df==False:
            print(f'''Model Performance
            ESS = {round(ess,5)}
            SSE = {round(sse,5)}
            TSS = {round(tss,5)}
            MSE = {round(mse,5)}
            RMSE = {round(rmse,5)}
            R^2 = {round(r2v2,10)}
            AdjR^2 = {round(adjR2,5)}''')
        

        else:
            df = pd.DataFrame()
        
            df ={
                'ESS' : round(ess,3),
                'SSE' : round(sse,3),
                'TSS' : round(tss,3),
                'MSE' : round(mse,3),
                'RMSE': round(rmse,3),
                'R^2' : round(r2v2,3),
                'AdjR^2':round(adjR2,3)
                }
                
            return df



def evaluate_models(y, yhat):

    ess, sse, tss, mse, rmse, r2 = calc_performance(y, yhat)

    df = pd.DataFrame()
    
    df ={
            'ESS' : round(ess,3),
            'SSE' : round(sse,3),
            'TSS' : round(tss,3),
            'MSE' : round(mse,3),
            'RMSE': round(rmse,3),
            'AdjR^2': round(r2,3)
        }
    
        
    return df



def better_than_baseline(y, yhat):
    SSE, ESS, TSS, MSE, RMSE = regression_errors(y, yhat)
    
    SSE_baseline, MSE_baseline, RMSE_baseline = baseline_mean_errors(y)
    
    if SSE < SSE_baseline:
        print('My OSL model performs better than baseline')
    else:
        print('My OSL model performs worse than baseline. :( )')



''' 
making models for each county


# la county
# fit data to simple regression
lm.fit(la_x_train, la_y_train)

# make predictions
la_simple_model = lm.predict(la_x_train)

# orange county
# fit data to simple regression
lm.fit(or_x_train, or_y_train)

# make predictions
or_simple_model = lm.predict(or_x_train)

# la county
# fit data to simple regression
lm.fit(vent_x_train, vent_y_train)

# make predictions
vent_simple_model = lm.predict(vent_x_train)

'''


import pandas as pd
import numpy as np


from sklearn.preprocessing import  PolynomialFeatures
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, explained_variance_score

# linear regressions
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor

# non-linear regressions
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor




############## GLOBAL VARIABLES ###########
seed = 42 # random seed for random_states
features = ['garage_sqft', 'age','beds','garage','fireplace','bath',\
            'bed_bath_ratio','lot_sqft','tax_amount','hottub_spa', 'Orange',\
            'Ventura', 'LA','logerror']
features_counties = ['garage_sqft', 'age','beds','garage','fireplace','bath',\
            'bed_bath_ratio','lot_sqft','tax_amount','hottub_spa', 'logerror']

# get zillow data
df = wr.get_zillow()

# separate data based on location
la_city = df[df.county_name == 'LA_city'] # LA city
la = df[df.county_name == 'LA'] # LA county
ventura = df[df.county_name == 'Ventura'] # Ventura county
orange = df[df.county_name == 'Orange'] # Orange county

# remove unneeded columns in counties data sets

la_city = la_city[features_counties]
la = la[features_counties]
ventura = ventura[features_counties]
orange = orange[features_counties]

# remove unneeded columns and add dummy variables for county_name in the main data set
df = wr.dummies(df)
df = df[features]

#split_counties into train, validate, test data sets and target vars
XLA1, XLA2, XLA3, yla1, yla2, yla3 = wr.full_split_zillow(la)
XLC1, XLC2, XLC3, ylc1, ylc2, ylc3 = wr.full_split_zillow(la_city)
XO1, XO2, XO3, yo1, yo2, yo3 = wr.full_split_zillow(ventura)
XV1, XV2, XV3, yv1, yv2, yv3 = wr.full_split_zillow(orange)
# scale counties data sets
XLA1, XLA2, XLA3 = wr.standard_scale_zillow(XLA1, XLA2, XLA3, counties=True)
XLC1, XLC2, XLC3 = wr.standard_scale_zillow(XLC1, XLC2, XLC3, counties=True)
XO1, XO2, XO3 = wr.standard_scale_zillow(XO1, XO2, XO3, counties=True)
XV1, XV2, XV3 = wr.standard_scale_zillow(XV1, XV2, XV3, counties=True)

# split the main data into 3 data sets and 3 target arrays
X_train, X_validate, X_test, y_train, y_validate, y_test = wr.full_split_zillow(df)

# get scaled X_train, X_validate, X_test sets
# standard scaler
X_train, X_validate, X_test = wr.standard_scale_zillow(X_train, X_validate, X_test)

# get a baseline value = median of the train set's target
baseline = y_train.mean()


###### GLOBAL EVALUATION VARS ##########

# DataFrame to keep model's evaluations
scores = pd.DataFrame(columns=['model_name', 'feature_name', 'R2_train', 'R2_validate'])

# create a dictionary of regression models
models = {
    'Linear Regression': LinearRegression(),
    'Gradient Boosting Regression': GradientBoostingRegressor(random_state=seed),
    'Decision Tree Regression': DecisionTreeRegressor(max_depth=4, random_state=seed),
    'Random Forest Regression':RandomForestRegressor(max_depth=4, random_state=seed),
    'LassoLars Regression':LassoLars(alpha=0.1)
    }


############### EVALUATION FUNCTIONS #############

def regression_errors(y_actual, y_predicted):
    '''
    returns r^2 score
    '''

    # adjucted R^2 score
    ADJR2 = explained_variance_score(y_actual, y_predicted)
    return round(ADJR2, 2)

############### MODELING FUNCTIONS ###############

def run_models(X_train, X_validate, y_train, y_validate, f_name='stand '):
    
    '''
    general function to run models with X_train and X_validate that were scaled
    '''
    feature_name = f_name # + str(f_number)
    for item in models:
        # create a model
        model = models[item]
        # fit the model
        model.fit(X_train, y_train)
        # predictions of the train set
        y_hat_train = model.predict(X_train)
        # predictions of the validate set
        y_hat_validate = model.predict(X_validate)


        # calculate scores train set
        R2 = regression_errors(y_train, y_hat_train)
        # calculate scores validation set
        R2_val = regression_errors(y_validate, y_hat_validate)
        
        
        # add the score results to the scores Data Frame
        scores.loc[len(scores.index)] = [item, feature_name, R2, R2_val]

def run_polinomial(X1, X2, y_train, y_validate, f_name='poly '):
    '''
    
    '''
    f = ['beds', 'bath']
    poly = PolynomialFeatures(degree=3, include_bias=False, interaction_only=False)
    poly.fit(X1[f])
    # create a df with transformed features of the train set
    X1_poly = pd.DataFrame(
        poly.transform(X1[f]),
        columns=poly.get_feature_names(f),
        index=X1.index)
    X1_poly = pd.concat([X1_poly, X1[f]], axis=1)
    #X1_poly = pd.concat([X1_poly, X1], axis=1)

    #display(X1_poly.head(1)) #testing the columns

    # create a df with transformed features for the validate set
    X2_poly = pd.DataFrame(
        poly.transform(X2[f]),
        columns=poly.get_feature_names(X2[f].columns),
        index=X2.index)
    X2_poly = pd.concat([X2_poly, X2[f]], axis=1)
    #X2_poly = pd.concat([X2_poly, X2], axis=1)

    feature_name = f_name #+ str(f_number)

    for key in models:
        # create a model
        model = models[key]
        # fit the model
        model.fit(X1_poly, y_train)
        # predictions of the train set
        y_hat_train = model.predict(X1_poly)
        # predictions of the validate set
        y_hat_validate = model.predict(X2_poly)

        # calculate scores train set
        R2 = regression_errors(y_train, y_hat_train)
        # calculate scores validation set
        R2_val = regression_errors(y_validate, y_hat_validate)
        

        # add the score results to the scores Data Frame
        scores.loc[len(scores.index)] = [key, feature_name, R2, R2_val]


def get_scores():
    # empty the scores data frame
    scores.drop(scores.index, axis=0, inplace=True)
    run_models(X_train, X_validate, y_train, y_validate, f_name='stand ')
    run_polinomial(X_train.iloc[:, :-1], X_validate.iloc[:, :-1], y_train, y_validate)
    return scores.sort_values(by=['R2_train'], ascending=False).head(10)


############# RUN MODELS ON CLUSTERS ##############

X_train_num, X_validate_num, X_test_num = cl.add_numerical_clusters(X_train, X_validate, X_test)
X_train_loc, X_validate_loc, X_test_loc = cl.add_location_clusters(X_train, X_validate, X_test)

def check_numerical_clusters():
    '''
    run models on numerical cluster data frames
    '''
    # empty the scores data frame
    scores.drop(scores.index, axis=0, inplace=True)
    
    # create data frames based on clusters and run models
    for j in range(6):
        # separate by clusters
        X1 = X_train_num[X_train_loc.numerical_clusters == j]
        X2 = X_validate_num[X_validate_loc.numerical_clusters == j]
        # drop column location_cluster
        X1.drop(columns='numerical_clusters', inplace=True)
        X2.drop(columns='numerical_clusters', inplace=True)
        # separate y_train
        y1 = y_train[X1.index]
        y2 = y_validate[X2.index]
        
        # run models
        run_models(X1, X2, y1, y2, j)
        run_polinomial(X1.iloc[:, :-1], X2.iloc[:, :-1], y1, y2, j)
        
    return scores.head(5)

def check_location_clusters():
    '''
    run models on location cluster data frames
    '''
    # empty the scores data frame
    scores.drop(scores.index, axis=0, inplace=True)
    
    # create data frames based on clusters and run models
    for j in range(6):
        # separate by clusters
        X1 = X_train_loc[X_train_loc.location_clusters == j]
        X2 = X_validate_loc[X_validate_loc.location_clusters == j]
        # drop column location_cluster
        X1.drop(columns='location_clusters', inplace=True)
        X2.drop(columns='location_clusters', inplace=True)
        # separate y_train
        y1 = y_train[X1.index]
        y2 = y_validate[X2.index]
        
        # run models
        run_models(X1, X2, y1, y2, j)
        run_polinomial(X1.iloc[:, :-1], X2.iloc[:, :-1], y1, y2, j)
        
    return scores.head(5)

def get_cluster_scores():
    '''
    this function runs models on clustered subsets
    returns the data frame with first 10 results of 
    '''
    loc_clust = check_location_clusters().iloc[:10, :]
    num_clust = check_numerical_clusters().iloc[:10, :]
    cluster_results = pd.concat([loc_clust, num_clust], axis=1)
    columns = ['location_clusters', 'feature_name_loc', 'Location_R2_train', 'R2_val_loc',\
           'numerical_clusters', 'feature_name_num', 'Numerical_R2_train', 'R2_val_num']
    cluster_results.columns = columns
    columns2 = ['Location_R2_train', 'R2_val_loc', 'Numerical_R2_train', 'R2_val_num']
    cluster_results = cluster_results[columns2]
    
    return cluster_results

######### RUN MODELS ON COUNTY DATA SETS
def get_counties_scores(): 
    # empty the scores data frame
    scores.drop(scores.index, axis=0, inplace=True)
    # la county
    run_models(XLA1, XLA2, yla1, yla2, f_name='la stand')
    run_polinomial(XLA1, XLA2, yla1, yla2, f_name='la poly')

    # la city
    run_models(XLC1, XLC2, ylc1, ylc2, f_name='la_city stand')
    run_polinomial(XLC1, XLC2, ylc1, ylc2, f_name='la_city poly')

    # orange county
    run_models(XO1, XO2, yo1, yo2, f_name='orange stand')
    run_polinomial(XO1, XO2, yo1, yo2, f_name='orange poly')

    # ventura county
    run_models(XV1, XV2, yv1, yv2, f_name='ventura stand')
    run_polinomial(XV1, XV2, yv1, yv2, f_name='ventura poly')
    
    return scores.sort_values(by=['R2_train', 'R2_validate'], ascending=False).head(10)

####### get the scores of the best model ###########
def get_final_scores():
    XLC1, XLC2, XLC3, ylc1, ylc2, ylc3 = wr.full_split_zillow(la_city)
    LC1, XLC2, XLC3 = wr.standard_scale_zillow(XLC1, XLC2, XLC3, counties=True)
    rf = RandomForestRegressor(max_depth=4, random_state=seed)
    rf.fit(XLC1, ylc1)
    y_hat_train = rf.predict(XLC1)
    y_hat_validate = rf.predict(XLC2)
    y_hat_test = rf.predict(XLC3)
    R2_train = regression_errors(ylc1, y_hat_train)
    R2_validate = regression_errors(ylc2, y_hat_validate)
    R2_test = regression_errors(ylc3, y_hat_test)
    final_scores = pd.DataFrame(columns=['model_name','train', 'validate', 'test'])
    final_scores.loc[len(final_scores.index)] = ['Random Forest Regressor', R2_train, R2_validate, R2_test]
    return final_scores