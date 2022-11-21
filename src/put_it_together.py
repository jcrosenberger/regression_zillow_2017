import src.wrangle as wr
import src.evaluate as ev 
import matplotlib.pyplot as plt
import seaborn as sns




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
    variables = ['tax_value', 'bedrooms', 'bath_adv', 'sq_feet', 'lot_size', 'year_built']
    n=1
    for i in variables:
        #n = 1
        plt.subplot(4,2,n)
        sns.boxplot(x = df[i])
        n +=1

    plt.subplots_adjust(hspace=1)   
    plt.suptitle('outlier detection\nComplex Model')
    plt.show()



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