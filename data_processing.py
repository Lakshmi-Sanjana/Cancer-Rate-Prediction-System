
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
