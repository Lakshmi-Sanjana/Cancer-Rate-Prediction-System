
from data_ingest import IngestData
from data_processing import (find_constant_columns,delete_constant_columns,find_columns_with_few_values,drop_and_fill)
from feature_engineering import bin_to_num,one_hot_encoding,cat_to_col

ingest_data=IngestData()
df=ingest_data.get_data("./ols-data/data/cancer_new.csv")

constant_columns=find_constant_columns(df)
print("Columns that contain a single value : ",constant_columns)

columns_with_few_values = find_columns_with_few_values(df,10)
print("Columns that contain few values : ",columns_with_few_values)

df=bin_to_num(df)
df=cat_to_col(df)
df=one_hot_encoding(df)
df=drop_and_fill(df)
print(df.shape)
df.to_csv("ols-data/cancer_reg_processed.csv",index=False)
