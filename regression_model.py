
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from data_processing import split_data

def correlation_among_numeric_features(df,cols):
    numeric_col=df[cols]
    corr=numeric_col.corr()
    # get highly correlated features and also tell to which fetaure it is correlated
    corr_features=set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i,j])>0.8:
                colname=corr.columns[i]
                corr_features.add(colname)
    return corr_features

def lr_model(x_train,y_train):
    #create a fitted model
    x_train_with_intercept=sm.add_constant(x_train)
    lr=sm.OLS(y_train,x_train_with_intercept).fit()
    return lr

def identify_significant_vars(lr,p_value_threshold=0.05):
    #print the p-values
    print(lr.pvalues)

    # print the r-squared value for the model
    print(lr.rsquared)

    #print the adjusted r-squared value for the model
    print(lr.rsquared_adj)
    
    #identify the significant variables
    significant_vars = [var for var in lr.pvalues.keys() if lr.pvalues[var]<p_value_threshold]
    return significant_vars



if __name__ == "__main__":
    capped_data= pd.read_csv("ols-data/capped_data.csv")
    print(capped_data)

    corr_features= correlation_among_numeric_features(capped_data,capped_data.columns)
    print(corr_features)
    
    

    cols = [col for col in capped_data.columns if col not in corr_features or col == "TARGET_deathRate"]

    len(cols)
    x_train,x_test,y_train,y_test= split_data(capped_data[cols],"TARGET_deathRate")
    lr=lr_model(x_train,y_train)
    summary=lr.summary()
    print(summary)



    significant_vars = identify_significant_vars(lr)
    print(len(significant_vars))

    #train the model with significant variables
    x_test_with_intercept = sm.add_constant(x_test)

# Make predictions
    y_pred = lr.predict(x_test_with_intercept)

# Evaluate the model
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

# Print metrics
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R-squared Score:", r2)

    import matplotlib.pyplot as plt

# Scatter plot
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")
    plt.show()

    # significant_vars.remove("const")
    # x_train=sm.add_constant(x_train)
    # lr=lr_model(x_train[significant_vars],y_train)
    # summary=lr.summary()
    # summary
