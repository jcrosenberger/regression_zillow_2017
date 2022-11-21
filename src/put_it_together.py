import src.wrangle as wr
import src.evaluate as ev 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

def explore_simple(df):
    plt.figure(figsize=(10,10))
    variables = ['tax_value', 'bedrooms', 'baths', 'sq_feet']
    n=1
    for i in variables:
        #n = 1
        plt.subplot(4,2,n)
        sns.boxplot(x = df[i])
        n +=1

    plt.subplots_adjust(hspace=1)   
    plt.suptitle('outlier detection\nSimple Model')
    plt.show()



def explore_complex(df):
    plt.figure(figsize=(10,10))
    variables = ['tax_value', 'bedrooms', 'bath_adv', 'squared_sq_feet', 'lot_size', 'year_built']
    n=1
    for i in variables:
        #n = 1
        plt.subplot(4,2,n)
        sns.boxplot(x = df[i])
        n +=1

    plt.subplots_adjust(hspace=1)   
    plt.suptitle('outlier detection\nComplex Model')
    plt.show()


def correlate_viz(df, df2, target):

    # sets size of the vizualization product
    plt.figure(figsize=(10,10))

    # DataFrame 1 - both features
    # creates a vertical heat map, correlating values in dataframe with a feature in the dataframe 
    # (the target value to be predicted)
    plt.subplot(1,2,1)
    heatmap = sns.heatmap(df.corr()[[target]].sort_values(by=target, ascending = False), vmin=-1, vmax=1, annot=True,cmap='BrBG')

    # title information
    heatmap.set_title('Simple Model \nFeatures Correlating with \nTax Value', fontdict={'fontsize':18}, pad=16);
    
    # DataFrame 2 - 
    plt.subplot(1,2,2)
    heatmap = sns.heatmap(df2.corr()[[target]].sort_values(by=target, ascending = False), vmin=-1, vmax=1, annot=True,cmap='BrBG')

    # title information
    heatmap.set_title('Complex Model \nFeatures Correlating with \nTax Value', fontdict={'fontsize':18}, pad=16);


def fips_viz(df):
    plt.figure(figsize=(20,20))
    big_variables= ['tax_value','sq_feet']
    small_variables= ['bedrooms','baths']

    n =220
    for i in big_variables:
        n+=1 
        plt.subplot(n)
        sns.histplot(data = df, x=i, hue='fips', kde = True, bins= 50, palette='hsv_r')

    for i in small_variables:
        n+=1 
        plt.subplot(n)
        sns.histplot(data = df, x=i, hue='fips', bins = 4, palette='hsv')    


    plt.show()


def spearman_test(df, target_var, test_var):
    r, p_value = spearmanr(df[target_var], df[test_var])
    print(f'Correlation Coefficient: {r}\nP-value: {p_value}')


def baseline_calc(df):

    x_train, y_train, x_validate, y_validate, x_test, y_test = wr.x_y(df, 'tax_value')
    # creating dataframe to hold values for comparison between prediction models
    predictions = pd.DataFrame()

    # target column is the variable we are trying to predict with machine learning
    predictions['target'] = y_train

    # produce baseline predictions based on the mean of the tax values and median values
    predictions['baseline_mean'] = int(y_train.mean())
    predictions['baseline_median'] = int(y_train.median())
    
    # creating dataframe to hold values for comparison between prediction models
    #predictions = pd.DataFrame()

    # creating simple regression model BEFORE splitting by county

    # make the model
    lm = LinearRegression()

    # fit data to simple regression
    lm.fit(x_train, y_train)

    # make predictions
    predictions['simple_model'] = lm.predict(x_train)

    predictions = predictions[np.isfinite(predictions).all(1)]

    return predictions



def simple_splits(df):
    '''
    Separating our big dataset by using boolean masks to identify our different counties

    '''

    keeper_variables = ['tax_value', 'bedrooms', 'baths', 'sq_feet']

    county = df['fips'] == 6037
    la_df = df[keeper_variables][county]

    county = df['fips'] == 6059
    orange_df = df[keeper_variables][county]

    county = df['fips'] == 6111
    ventura_df = df[keeper_variables][county]

    '''
    Preparing to analyze our simple model as asked for initial evaluatation to be made upon
    separating our data frames into train, validate and test data frames
    further separates our data into our indepdendent and dependent variables
    '''


    return la_df, orange_df, ventura_df



def measure_performance(df):
    df1 = pd.DataFrame()
    for i in df.columns:
        df1 = df1.append(ev.regression_errors(df['target'], df[i], df=True, features=4), ignore_index=True)
        

  
    df1['Model'] = np.array(df.columns)
    df1.set_index('Model', inplace=True)
    return df1




def simple_regression_workhorse(model_df):

    lm = LinearRegression()
    
    keeper_variables = ['tax_value', 'bedrooms', 'baths', 'sq_feet']

    county = model_df['fips'] == 6037
    la_df = model_df[keeper_variables][county]

    county = model_df['fips'] == 6059
    orange_df = model_df[keeper_variables][county]

    county = model_df['fips'] == 6111
    ventura_df = model_df[keeper_variables][county]



    # split la county dfs
    la_x_train, la_y_train, la_x_validate, la_y_validate, la_x_test, la_y_test = wr.x_y(la_df, 'tax_value')

    #split orange county dfs
    or_x_train, or_y_train, or_x_validate, or_y_validate, or_x_test, or_y_test = wr.x_y(orange_df, 'tax_value')

    # split ventura county dfs
    vent_x_train, vent_y_train, vent_x_validate, vent_y_validate, vent_x_test, vent_y_test = wr.x_y(ventura_df, 'tax_value')
    
    ''' 
    making models for each county
    '''

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
    running the initial set of models through the function which will create our initial dataframe
    holding performance measures for each model
    '''
    predictions = baseline_calc(model_df)
    comparison_df = measure_performance(predictions)
    
    
    ''' 
    appending to the performance measure recording dataframe, our new models' performance
    '''
    comparison_df = comparison_df.append(ev.regression_errors(la_y_train, la_simple_model, df=True, features = 5), ignore_index=True)
    comparison_df = comparison_df.append(ev.regression_errors(or_y_train, or_simple_model, df=True, features = 5), ignore_index=True)
    comparison_df = comparison_df.append(ev.regression_errors(vent_y_train, vent_simple_model, df=True, features = 5), ignore_index=True)
    
    '''
    replacing the index with the name of the models in our performance measure recording dataframe
    '''

    model_list = ['target', 'baseline_mean', 'baseline_median', 
                  'simple_model', 'la_simple_model', 'or_simple_model', 'vent_simple_model']

    comparison_df['Model'] = model_list
    comparison_df.set_index('Model', inplace=True)
    
    
    return comparison_df




def complex_regression_workhorse(df, compare_df):

    lm = LinearRegression()
    
    keeper_variables = ['tax_value', 'bedrooms', 'bath_adv', 'squared_sq_feet', 'lot_size', 'year_built']

    county = df['fips'] == 6037
    la_df = df[keeper_variables][county]

    county = df['fips'] == 6059
    orange_df = df[keeper_variables][county]

    county = df['fips'] == 6111
    ventura_df = df[keeper_variables][county]

    
    # split la county dfs
    la_x_train, la_y_train, la_x_validate, la_y_validate, la_x_test, la_y_test = wr.x_y(la_df, 'tax_value')

    #split orange county dfs
    or_x_train, or_y_train, or_x_validate, or_y_validate, or_x_test, or_y_test = wr.x_y(orange_df, 'tax_value')

    # split ventura county dfs
    vent_x_train, vent_y_train, vent_x_validate, vent_y_validate, vent_x_test, vent_y_test = wr.x_y(ventura_df, 'tax_value')
    
    ''' 
    making models for each county
    '''

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
    running the initial set of models through the function which will create our initial dataframe
    holding performance measures for each model
    '''

    comparison_df = compare_df
    
    
    ''' 
    appending to the performance measure recording dataframe, our new models' performance
    '''
    comparison_df = comparison_df.append(ev.regression_errors(la_y_train, la_simple_model, df=True, features = 6), ignore_index=True)
    comparison_df = comparison_df.append(ev.regression_errors(or_y_train, or_simple_model, df=True, features = 6), ignore_index=True)
    comparison_df = comparison_df.append(ev.regression_errors(vent_y_train, vent_simple_model, df=True, features = 6), ignore_index=True)
    
    '''
    replacing the index with the name of the models in our performance measure recording dataframe
    '''

    model_list = ['target', 'baseline_mean', 'baseline_median', 
                  'simple_model', 'la_simple_model', 'or_simple_model', 'vent_simple_model',
                 'la_complex_model', 'or_complex_model', 'vent_complex_model']

    comparison_df['Model'] = model_list
    comparison_df.set_index('Model', inplace=True)
    
    
    return comparison_df




def pie_chart1(comparison_df):
    '''
    Pie chart for Model 1
    '''

    labels = ['Explained portion', 'Unexplained Portion']
    values = [comparison_df['AdjR^2']['simple_model'],(1-comparison_df['AdjR^2']['simple_model'])]
    explode = [0.05, 0]
    palette_color = sns.color_palette('tab10')

    #setting title information
    plt.title('Simplest Model Performance using Adjusted R^2 \nMore Blue is Better')

    #creating pie with variables set above
    plt.pie(values, labels=labels, colors = palette_color, explode = explode, autopct='%.0f%%')

    plt.show()



def overload_pies(comparison_df):
    '''
    Pie charts for County Specific Models
    '''
    plt.figure(figsize=(15,10))

    models = ['baseline_mean', 'simple_model',
         'la_simple_model', 'or_simple_model', 'vent_simple_model',
         'la_complex_model', 'or_complex_model', 'vent_complex_model']
    labels = ['Explained', 'Unexplained']
    n=2
    for i in models:
        plt.subplot(3,3,n)
        values = [comparison_df['AdjR^2'][i],(1-comparison_df['AdjR^2'][i])]
        explode = [0.05, 0]
        palette_color = sns.color_palette('tab10')

        #setting title information
        plt.title(f'{i}')

        #creating pie with variables set above
        plt.pie(values, labels=labels, colors = palette_color, explode = explode, autopct='%.0f%%')
        n+=1

    plt.suptitle('Model Performance Using Adjusted R^2 \n More Blue is better')
    plt.show()


def final_pies(comparison_df):
    '''
    Pie charts for County Specific Models
    '''
    plt.figure(figsize=(15,10))

    models = ['la_complex_model', 'or_complex_model', 'vent_complex_model']
    labels = ['Explained', 'Unexplained']
    n=1
    for i in models:
        plt.subplot(1,3,n)
        values = [comparison_df['AdjR^2'][i],(1-comparison_df['AdjR^2'][i])]
        explode = [0.05, 0]
        palette_color = sns.color_palette('tab10')

        #setting title information
        plt.title(f'{i}')

        #creating pie with variables set above
        plt.pie(values, labels=labels, colors = palette_color, explode = explode, autopct='%.0f%%')
        n+=1

    plt.suptitle('Model Performance Using Adjusted R^2 \n More Blue is better')
    plt.show()



def test_model(df, baseline_model_df):
    predictions = baseline_calc(df)
    
    lm = LinearRegression()
    
    keeper_variables = ['tax_value', 'bedrooms', 'bath_adv', 'squared_sq_feet', 'lot_size', 'year_built']

    county = df['fips'] == 6037
    la_df = df[keeper_variables][county]

    county = df['fips'] == 6059
    orange_df = df[keeper_variables][county]

    county = df['fips'] == 6111
    ventura_df = df[keeper_variables][county]
    
    
    # split la county dfs
    la_x_train, la_y_train, la_x_validate, la_y_validate, la_x_test, la_y_test = wr.x_y(la_df, 'tax_value')

    #split orange county dfs
    or_x_train, or_y_train, or_x_validate, or_y_validate, or_x_test, or_y_test = wr.x_y(orange_df, 'tax_value')

    # split ventura county dfs
    vent_x_train, vent_y_train, vent_x_validate, vent_y_validate, vent_x_test, vent_y_test = wr.x_y(ventura_df, 'tax_value')
    
    
    # la county
    # fit data to simple regression
    lm.fit(la_x_test, la_y_test)

    # make predictions
    la_complex_model = lm.predict(la_x_test)

    # orange county
    # fit data to simple regression
    lm.fit(or_x_test, or_y_test)

    # make predictions
    or_complex_model = lm.predict(or_x_test)

    # la county
    # fit data to simple regression
    lm.fit(vent_x_test, vent_y_test)

    # make predictions
    vent_complex_model = lm.predict(vent_x_test)
    
    '''
    running the initial set of models through the function which will create our initial dataframe
    holding performance measures for each model
    '''

    #comparison_df = pd.DataFrame
    predictions = baseline_calc(baseline_model_df)
    comparison_df = measure_performance(predictions)
    
    
    ''' 
    appending to the performance measure recording dataframe, our new models' performance
    '''
    comparison_df = comparison_df.append(ev.regression_errors(la_y_test, la_complex_model, df=True, features = 6), ignore_index=True)
    comparison_df = comparison_df.append(ev.regression_errors(or_y_test, or_complex_model, df=True, features = 6), ignore_index=True)
    comparison_df = comparison_df.append(ev.regression_errors(vent_y_test, vent_complex_model, df=True, features = 6), ignore_index=True)
    
    '''
    replacing the index with the name of the models in our performance measure recording dataframe
    '''

    model_list = ['target', 'baseline_mean', 'baseline_median', 'simple_model',
        'la_complex_model', 'or_complex_model', 'vent_complex_model']

    comparison_df['Model'] = model_list
    comparison_df.set_index('Model', inplace=True)
    
    return comparison_df