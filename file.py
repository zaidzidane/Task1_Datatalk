import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
df=pd.read_parquet(r'C:\Users\mzaid\Downloads\yellow_tripdata_2022-01.parquet')
print(len(df.columns))# No. of columns Question 1
df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
print(df['duration'].describe()) #Question 2
df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
val=len(df)
df = df[(df.duration >= 1) & (df.duration <= 60)]
print(len(df)/val) #records left  Question 3

categorical = ['PULocationID', 'DOLocationID']
numerical = ['trip_distance']

df[categorical] = df[categorical].astype(str)
train_dicts = df[categorical + numerical].to_dict(orient='records')

dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)
print(X_train)
target = 'duration'
y_train = df[target].values

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_train)
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_train, y_pred, squared=False))

def read_dataframe(filename):
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)

        df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_pickup_datetime)
        df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)
    elif filename.endswith('.parquet'):
        df = pd.read_parquet(filename)

    df['duration'] = df.tpep_pickup_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df
df_train =read_dataframe(r'C:\Users\mzaid\Downloads\yellow_tripdata_2022-01.parquet')
df_val = read_dataframe(r'C:\Users\mzaid\Downloads\yellow_tripdata_2022-02.parquet')
df_train['PU_DO'] = df_train['PULocationID'] + '_' + df_train['DOLocationID']
df_val['PU_DO'] = df_val['PULocationID'] + '_' + df_val['DOLocationID']
categorical = ['PU_DO'] #'PULocationID', 'DOLocationID']
numerical = ['trip_distance']

dv = DictVectorizer()

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dicts)

target = 'duration'
y_train = df_train[target].values
y_val = df_val[target].values

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_val)

print("val Score",mean_squared_error(y_val, y_pred, squared=False))