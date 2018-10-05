import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')


def load_balancing(train, labels):
    print('\n> Load Balancing Resampling')
    print('\t- Original dataset shape {}'.format(Counter(labels)))

    imputer = Imputer(strategy='median')
    # train = pd.get_dummies(train)
    train = imputer.fit_transform(train)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_sample(train, labels)

    print('\t- Resampled dataset shape {}'.format(Counter(y_res)))
    print('\n> End of Load Balancing Resampling')
    return X_res, y_res


def memory_reduce(dataframe):
    print('\t> Launch Memory Reduction')

    #--- Memory usage of entire dataframe ---
    mem = dataframe.memory_usage(index=True).sum()
    print("\t\t- Initial size {:.2f} MB".format(mem/ 1024**2))

    #--- List of columns that cannot be reduced in terms of memory size ---
    count = 0
    for c in dataframe.columns:
        if dataframe[c].dtype == object:
            count+=1
    print('\t\t- There are {} columns that cannot be reduced'.format(count))

    count = 0
    for c in dataframe.columns:

        if dataframe[c].dtype in ['int8', 'int16', 'int32', 'int64']:
            
            if (np.iinfo(np.int8).min < dataframe[c].min()) and (dataframe[c].max() < np.iinfo(np.int8).max):
                count+=1
                dataframe[c] = dataframe[c].fillna(0).astype(np.int8)
            
            if (np.iinfo(np.int16).min < dataframe[c].min()) and (dataframe[c].max() < np.iinfo(np.int16).max) and ((np.iinfo(np.int8).min > dataframe[c].min()) or (dataframe[c].max() > np.iinfo(np.int8).max)):
                count+=1
                dataframe[c] = dataframe[c].fillna(0).astype(np.int16)
            
            if (np.iinfo(np.int32).min < dataframe[c].min()) and (dataframe[c].max() < np.iinfo(np.int32).max) and ((np.iinfo(np.int16).min > dataframe[c].min()) or (dataframe[c].max() > np.iinfo(np.int16).max)):
                count+=1
                dataframe[c] = dataframe[c].fillna(0).astype(np.int32)
            
            if (np.iinfo(np.int64).min < dataframe[c].min()) and (dataframe[c].max() < np.iinfo(np.int64).max) and ((np.iinfo(np.int32).min > dataframe[c].min()) or (dataframe[c].max() > np.iinfo(np.int32).max)):
                count+=1
                dataframe[c] = dataframe[c].fillna(0).astype(np.int64)

        if dataframe[c].dtype in ['float16', 'float32', 'float64']:
            
            if (np.finfo(np.float16).min < dataframe[c].min()) and (dataframe[c].max() < np.finfo(np.float16).max):
                count+=1
                dataframe[c] = dataframe[c].fillna(0).astype(np.float16)
            
            if (np.finfo(np.float32).min < dataframe[c].min()) and (dataframe[c].max() < np.finfo(np.float32).max) and ((np.finfo(np.float16).min > dataframe[c].min()) or (dataframe[c].max() > np.finfo(np.float16).max)):
                count+=1
                dataframe[c] = dataframe[c].fillna(0).astype(np.float32)
            
            if (np.finfo(np.float64).min < dataframe[c].min()) and (dataframe[c].max() < np.finfo(np.float64).max) and ((np.finfo(np.float32).min > dataframe[c].min()) or (dataframe[c].max() > np.finfo(np.float32).max)):
                count+=1
                dataframe[c] = dataframe[c].fillna(0).astype(np.float64)

    print('\t\t- There are {} columns reduced'.format(count))

    #--- Let us check the memory consumed again ---
    mem = dataframe.memory_usage(index=True).sum()
    print("\t\t- Final size {:.2f} MB".format(mem/ 1024**2))
    print('\t> End of Memory Reduction')
    return dataframe


def missing_ratio(dataframe, n=30, plot=True) :
    
    '''
    Compute the ratio of missing values by column and plot the latter

    Options : plot = True to display plot or False to disable plotting

    Returns the missing ratio dataframe

    '''
    try :

        all_data_na = (dataframe.isnull().sum() / len(dataframe)) * 100
        all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:n]
        missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

        if plot:
            f, ax = plt.subplots(figsize=(15, 12))
            plt.xticks(rotation='90')
            sns.barplot(x=all_data_na.index, y=all_data_na)
            plt.xlabel('Features', fontsize=15)
            plt.ylabel('Percent of missing values', fontsize=15)
            plt.title('Percent missing data by feature', fontsize=15)

        return(missing_data)

    except ValueError as e :

        print("The dataframe has no missing values, ", e)
