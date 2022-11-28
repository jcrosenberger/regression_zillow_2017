# Predicting Home Value

## Description

### Provided a sample prompt, produce a report predicting Home Values using Zillow Home Data from 2017

***You are a junior data scientist on the Zillow data science team and recieve the following email in your inbox:*** 

    We want to be able to predict the values of single unit properties that the tax district assesses using the property data from those whose last transaction was during the "hot months" (in terms of real estate demand) of May and June in 2017. We also need some additional information outside of the model.

    Zach lost the email that told us where these properties were located. Ugh, Zach :-/. Because property taxes are assessed at the county level, we would like to know what states and counties these are located in.

    We'd also like to know the distribution of tax rates for each county.

    The data should have the tax amounts and tax value of the home, so it shouldn't be too hard to calculate. Please include in your report to us the distribution of tax rates for each county so that we can see how much they vary within the properties in the county and the rates the bulk of the properties sit around.

    Note that this is separate from the model you will build, because if you use tax amount in your model, you would be using a future data point to predict a future data point, and that is cheating! In other words, for prediction purposes, we won't know tax amount until we know tax value.

    -- The Zillow Data Science Team


## Audience
- The Zillow Data Science Team.
- Produce a concise presentation in a jupyter notebook for your peers representing the efficacy of data science models to predict home values using annual tax accessed values as the proxy for home values to potential buyers.

## Deliverables
- Provide a Github repo containing a readme (thanks for taking a look!), a final report in a jupyter notebook, python modules to import and prepare data for analysis, supplemental artifacts of work in the form of exploratory and modeling notebooks


## Data Dictionary
**For the Simple Model**
| Column | Description |
| --- | ---|
| baths | Number of Bathrooms |
| bedrooms | Number of Bedrooms |
| sq_feet | Total finished square feet of home |
| fips | Broken down below. Fips represents the code for the county where a house resides |
| fips: 6037 | Los Angeles County |
| fips: 6059 | Orange County |
| fips: 6111 | Ventura County |
| tax_value | Proxy for the home's value to potential buyers |


**For the Complex Model**
| Column | Description |
| --- | ---|
| bath_adv | Number of Bathrooms, including partial bathrooms |
| bedrooms | Number of Bedrooms |
| lot size | The size of the lot on which the house sits |
| squared_sq_feet| The square feet of a home squared again to give the variable polynomial features |
| fips | Broken down below. Fips represents the code for the county where a house resides |
| fips: 6037 | Los Angeles County |
| fips: 6059 | Orange County |
| fips: 6111 | Ventura County |
| tax_value | Proxy for the home's value to potential buyers |


*The simple model represented above is broken into three models based on county. The features of the models were chosen by the data science team to demonstrate the predictive power of a simple model at predicting a home's value to prospective consumers.* 


## Reproduction of this work

### Clone this repo

### Modules

- By design, I put a library of modules in the src folder.

- The wrangle.py module has a dependency in the env module which must operate correctly. 
> you will acquire two dataframes using wr.zillow_2017
- A module named env.py is required to function properly. That module must contain the following function:  
>  *Must have credentials which grant access to the web facing codeup sql server.*    
    
    def get_db_url(df):

    url = f'mysql+pymysql://{user}:{password}@{host}/{df}'
    return url 
    
> *Additionally in the env.py module, user, password, and host must be defined for the function to work properly*

