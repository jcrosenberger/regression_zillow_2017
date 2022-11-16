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
from sklearn.metrics import mean_squared_error, r2_score



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
    r2V2 = r2_score(y, yhat)
    
    if featureN > 2:
        # Adjusted R Squared
        AdjR2= 1-(1-r2)*(len(y)-1)/(len(y)-featureN-1)

        return ess, sse, tss, mse, rmse, AdjR2    
    
    else:
        return ess, sse, tss, mse, rmse, r2, r2V2



def regression_errors(y, yhat, df=False, features=2):
    '''
    This module does the legwork for evaluating model efficacy. 
    The default argument 'df' will determine whether to pass a data frame with a dictionary 
    when called or to print the evaluation metrics. 
    The AdjR2Feature default argument allows for tuning an Adjusted R^2 value, which is more accurate
    than R^2 for models with multiple features/variables 

    '''
    ess, sse, tss, mse, rmse, r2, r2v2 = calc_performance(y, yhat, features)
    
    
    if df==False:
        print(f'''Model Performance
        ESS = {round(ess,5)}
        SSE = {round(sse,5)}
        TSS = {round(tss,5)}
        MSE = {round(mse,5)}
        RMSE = {round(rmse,5)}
        R^2 = {round(r2,10)}''')
    

    else:
        df = pd.DataFrame()
    
        df ={
            'ESS' : round(ess,3),
            'SSE' : round(sse,3),
            'TSS' : round(tss,3),
            'MSE' : round(mse,3),
            'RMSE': round(rmse,3),
            'R^2': round(r2,3),
            'R2V2': round(r2v2,3)
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