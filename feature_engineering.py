
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def bin_to_num(data):
    binnedinc=[]
    for i in data["binnedInc"]:
        i= i.strip("()[]")
        
        i=i.split(",")
        
        i=tuple(i)
        #convert list to tuple
        i=tuple(map(float,i))
        #convert the tuple to a list
        i=list(i)
        binnedinc.append(i)
    data["binnedInc"] = binnedinc
    #make a new column upper and lower bound
    data["lower_bound"]=[i[0] for i in data["binnedInc"]]
    data["upper_bound"]=[i[1] for i in data["binnedInc"]]
    #also find median
    data["median"]=(data["lower_bound"]+data["upper_bound"])/2
    #drop the binnedInc column
    data.drop("binnedInc",axis=1,inplace=True)
    return data

def cat_to_col(data):

    #make a new column for both state and country
    data["county"]=[i.split(",")[0] for i in data["Geography"]]
    data["state"]=[i.split(",")[1] for i in data["Geography"]]

    #drop the column
    data.drop("Geography",axis=1,inplace=True)
    return data

def one_hot_encoding(X):

    #select categorical columns
    categorical_columns=X.select_dtypes(include=["object"]).columns
    #one hot encode the categorical columns

    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    one_hot_encoded=one_hot_encoder.fit_transform(X[categorical_columns])

    #convert the one hot encoded into data frame

    one_hot_encoded=pd.DataFrame(
        one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(categorical_columns)
    )
    #drop the categorical columns from the original datafram
    X=X.drop(categorical_columns,axis=1)
    # concatenate the one hot encoded datafram to the original dataframes
    X=pd.concat([X,one_hot_encoded],axis=1)

    return X
from sklearn.model_selection import train_test_split as tts

def find_constant_columns(dataframe):

    """
        takes a data frame and returns columsn with only single value which basically are constant 
    """

    constant_columns=[]
    for column in dataframe.columns:
        
        unique_values= dataframe[column].unique()

        if(len(unique_values))==1:
            constant_columns.append(column)
    return constant_columns     

def delete_constant_columns(dataframe,columns_to_delete):

    dataframe=dataframe.drop(columns_to_delete,axis=1)
    return dataframe


def find_columns_with_few_values(dataframe,threshold):

    few_values_columns=[]
    for column in dataframe.columns:
        
        unique_values_count=len(dataframe[column].unique())

        if unique_values_count < threshold:
            few_values_columns.append(column)
    return few_values_columns       

def find_duplicate_rows(dataframe):

    duplicate_rows=dataframe[dataframe.duplicated()]
    return duplicate_rows


def delete_duplicate_rows(dataframe):

    dataframe=dataframe.drop_duplicates(keep="first")
    return dataframe


def drop_and_fill(dataframe):

    #get the columns with more than 50% missing values
    cols_to_drop=dataframe.columns[dataframe.isnull().mean()>0.5]
    #drop the columns
    dataframe=dataframe.drop(cols_to_drop,axis=1)

    #fill the remaining columsn which are having missing values less than half with the mean it self

    dataframe=dataframe.fillna(dataframe.mean())

    return dataframe



def split_data(data, target_col, test_size=0.2, random_state=42):
   
    X = data.drop(target_col,axis=1)  # Features (drop the target column)
    y = data[target_col]                # Target column
    
    # Split the data
    X_train, X_test, y_train, y_test = tts(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
