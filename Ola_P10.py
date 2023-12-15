# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:39:34 2023

@author: venki
"""

########### Ola bike Rides demand forecast ############
import pandas as pd
import numpy as np

# Load & Read the data set.

df = pd.read_csv('D:\\Project_Pro\\ML_Path_Moderate\\Project_10\\raw_data.csv', low_memory = False, compression = 'gzip' )

df.head()

"A customer_ID number at a particular timestamp can only have one entry "
# Remvoing Duplicate Entries('its', 'number') columns.
df.duplicated().sum() ## We have total duplicates(65860)
df[df.duplicated(subset = ['ts','number'], keep = False)]
# Duplicate entries is 113540

## Keeping first occurances
df.drop_duplicates(subset = ['ts', 'number'], inplace = True,  keep = 'last')

df.reset_index(inplace = True, drop  = True)

df.info()

# Missing values
df.isnull().sum() # number is showing object, i.e null values is 0

df['number'] = pd.to_numeric(df['number'], errors = 'coerce')

df.isnull().sum() ## 116 Nan rows

df.dropna(inplace = True)
len(df)  # 8315382
df.shape  # (8315382, 6)

df['number'] = pd.to_numeric(df['number'], errors = 'coerce', downcast = 'integer')
df['ts'] = pd.to_datetime(df['ts'])

# Info of Dataset
df.info()

# Breaking time to features
df['hour'] = df['ts'].dt.hour
df['mins'] = df['ts'].dt.minute
df['day']  = df['ts'].dt.day
df['month'] = df['ts'].dt.month
df['year'] = df['ts'].dt.year
df['dayofweek'] = df['ts'].dt.dayofweek
df.columns

df.to_csv('D:\\Project_Pro\\ML_Path_Moderate\\Project_10\\preprocessed_1.csv', index = False, compression = 'gzip')
df

################ Data cleaning ##################
"""There can be cases when auser requests a ride and their booking requests is logged in our database
butnthis user re-books his/her ride due to longer wait hours or driver refused booking or user by mistakes
added wrong pickup or drop locations.
Case1: rebooking again to same location; Keep only one request of same user to same pickup lattitude longitude in 1hour time
 frame of first ride request.
 Case2: Location entry mistake; Keep only last request of user within 8 mintues of first booking request.
 A person booking a ride would generally book a ride that would take 8 mins of bike ride time.
 Also, Calculate distance b/w pickup and drop. based on distance and request time different remove bad data entries.
 Case2.1: Pick Up and Drop Lat-Long Distance less that 50 meters = 0.05kms: 
     no user do not like to ride just 50meters.
 Case3: Booking Location Outside operation zone of OLABikes"""
 
df = pd.read_csv('D:\\Project_Pro\\ML_Path_Moderate\\Project_10\\preprocessed_1.csv', compression = 'gzip')
 
#Check lat-long buling box coordinates
df['ts'] = pd.to_datetime(df['ts'])
df.sort_values(by = ['number', 'ts'], inplace = True)
df.reset_index(inplace = True)

df['booking_timestamp'] = df.ts.values.astype(np.int64)//10**9
df.head()

df['shift_booking_ts'] = df.groupby('number')['booking_timestamp'].shift(1)
df['shift_booking_ts'].fillna(0, inplace = True)

df['shift_booking_ts'] = df['shift_booking_ts'].astype('int64')

df['booking_time_diff_hr'] = round((df['booking_timestamp'] - df['shift_booking_ts'])//3600)
df['booking_time_diff_min'] = round((df['booking_timestamp'] - df['shift_booking_ts'])//60)

# Booking time differnce in min
df['booking_time_diff_min'].value_counts().to_dict()

len(df)
# Booking time differnce in hr
df['booking_time_diff_hr'].value_counts().to_dict()

df.shape # (8315382, 17)

# Case1; Rebooking again to same location within 1 hour by same user.
df = df[~((df.duplicated(subset = ['number','pick_lat', 'pick_lng'], keep = False)) & (df.booking_time_diff_hr <= 1))]
len(df) #  we have 4335944, and 3979554 rows are removed.

df.to_csv('D:\\Project_Pro\\ML_Path_Moderate\\Project_10\\preprocessed_2.csv', index = False, compression = 'gzip')


#Case2; One user books rides are different kat-long with in 8min time (ride time + driver arrival time)
# fraud User
# Human error booking
df = pd.read_csv('D:\\Project_Pro\\ML_Path_Moderate\\Project_10\\preprocessed_2.csv', compression = 'gzip')

print("Number of rides booked by same customer within  8mins time:{}".format(len(df[(df.booking_time_diff_min < 8)])))
# Number of rides booked by same customer within 8 mins time:611891

print("Number of rides booked by same customer morethan 8min time : {}".format(len(df[(df.booking_time_diff_min>=8)])))
# Number of rides booked by same customer morethan 8min time : 3724053

# Check Learning Resources Folder in Documentation Folder
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import time

def geodestic_distance(pick_lat, pick_lng, drop_lat, drop_lng):
    # 1mile = 1.60934 Kms
    return round(geodesic((pick_lat, pick_lng), (drop_lat, drop_lng)).miles*1.60934,2)

df['geodesic_distance'] = np.vectorize(geodestic_distance)(df['pick_lat'],df['pick_lng'],df['drop_lat'],df['drop_lng'])

## Number of rides booked but same customer within 8 mins time:875816

df[df['geodesic_distance'] <= 0.5]['geodesic_distance'].value_counts()

### Handle Case 2.1: Removing ride request less than 0.05 miles = 50 meters.

print("Number of Rides Requests less than 50meters: {}".format(len(df[df.geodesic_distance<=0.05])))
# number of Rides Requests less than 50meters : s14460
df = df[df.geodesic_distance>0.05]

len(df)  #3709593

df.to_csv('D:\\Project_Pro\\ML_Path_Moderate\\Project_10\\preprocessed_3.csv', index = False, compression = 'gzip')

# Case3: Rides request in non-operational regions.
# OLA Bikes OPERATIONS CITY(Bangalore)

geolocator = Nominatim(user_agent="OLABikes")
df = pd.read_csv('D:\\Project_Pro\\ML_Path_Moderate\\Project_10\\preprocessed_3.csv', compression = 'gzip')
location = geolocator.geocode("India")
location.raw
df
## How many rides outside india?
df[(df.pick_lat<=6.2325274) | (df.pick_lat>=35.6745457) | (df.pick_lng<=68.1113787) | (df.pick_lng>=97.395561) | (df.drop_lat<=6.2325274) | (df.drop_lat>=35.6745457) | (df.drop_lng<=68.1113787) | (df.drop_lng>=97.395561)]

# OLa bikes is only operational in India
# Removing all rides for which pickup or drop is outside INDIA

df.reset_index(inplace = True, drop = True)
outside_India = df[(df.pick_lat<=6.2325274) | (df.pick_lat>=35.6745457) | (df.pick_lng<=68.1113787) | (df.pick_lng>=97.395561) | (df.drop_lat<=6.2325274) | (df.drop_lat>=35.6745457) | (df.drop_lng<=68.1113787) | (df.drop_lng>=97.395561)]
df = df[~df.index.isin(outside_India.index)].reset_index(drop = True)

print("Number of Good Ride Requests: {}".format(len(df)))
#Number of Good Ride Requests: 3708951

## How many pickups and drops are outside bangalore?
pck_outside_bng = df[(df.pick_lat<=12.8340125) | (df.pick_lat>=13.1436649) | (df.pick_lng<=77.4601025) | (df.pick_lng>=77.7840515)]
drp_outside_bng = df[(df.drop_lat<=12.8340125) | (df.drop_lat>=13.1436649) | (df.drop_lng<=77.4601025) | (df.drop_lng>=77.7840515)]
print("Number of Pickup Requests Outside Bangalore: ",len(pck_outside_bng))
print("Number of Customers pickup outside Bangalore: ",len(np.unique(pck_outside_bng['number'].values)))

print("Number of Drops Requests Outside Bangalore: ",len(drp_outside_bng))
print("Number of Customers Drop outside Bangalore: ",len(np.unique(drp_outside_bng['number'].values)))

### Bounding PickUp Lat-Long Within State Karnataka
# ['11.5945587', '18.4767308', '74.0543908', '78.588083']
pck_outside_KA = df[(df.pick_lat<=11.5945587) | (df.pick_lat>=18.4767308) | (df.pick_lng<=74.0543908) | (df.pick_lng>=78.588083)]
drp_outside_KA = df[(df.drop_lat<=11.5945587) | (df.drop_lat>=18.4767308) | (df.drop_lng<=74.0543908) | (df.drop_lng>=78.588083)]
print("Pickups Outisde KA: {} \nDrop outside KA: {}".format(len(pck_outside_KA),len(drp_outside_KA)))
print("Number of Customers Drop outside KA: ",len(np.unique(drp_outside_KA['number'].values)))
print("Number of Customers pickup outside KA: ",len(np.unique(pck_outside_KA['number'].values)))

total_ride_outside_KA = df[(df.pick_lat<=11.5945587) | (df.pick_lat>=18.4767308) | (df.pick_lng<=74.0543908) | (df.pick_lng>=78.588083) | (df.drop_lat<=11.5945587) | (df.drop_lat>=18.4767308) | (df.drop_lng<=74.0543908) | (df.drop_lng>=78.588083)]
print("Total Ride Outside Karnataka: {}".format(len(total_ride_outside_KA)))
# Total Ride Outside Karnataka: 39632

# Ola Bikes doesnot provide intercity request. Considering these as system error requests.
## Rides for which geodesic distance > 500kms
## Pickup and drop not of KA (state where we have maximum booking requests and user base)
suspected_bad_rides = total_ride_outside_KA[total_ride_outside_KA.geodesic_distance > 500]
suspected_bad_rides
## these are bad rides(506), so we can drop it

df = df[~df.index.isin(suspected_bad_rides.index)].reset_index(drop = True)

print("Number of Good Ride Requests: {}".format(len(df))) #3708445

dataset = df[['ts', 'number', 'pick_lat','pick_lng','drop_lat','drop_lng','geodesic_distance','hour','mins','day','month','year','dayofweek','booking_timestamp','booking_time_diff_hr', 'booking_time_diff_min']]

dataset.to_csv('D:\\Project_Pro\\ML_Path_Moderate\\Project_10\\clean1_data.csv', index = False, compression = 'gzip')

import pandas as pd
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from sklearn.cluster import MiniBatchKMeans, KMeans
import gpxpy.geo
from datetime import datetime, timedelta
from joblib import dump, load
import pandas_profiling

df = pd.read_csv('D:\\Project_Pro\\ML_Path_Moderate\\Project_10\\clean_data.csv', compression = 'gzip')
len(df)  ## 3708329
df.columns
## Geospacial Feature Engineering - Clustering/Segmentation
#We have divided whole india into regions using 'K-Means Clustering'
coord = df[['pick_lat', 'pick_lng']].values
neighbors = []

def min_distance(regionCenters, totalClusters):
    good_points = 0
    bad_points = 0
    less_dist = []
    more_dist = []
    min_distance = np.inf  #any big number can be given here
    for i in range(totalClusters):
        good_points = 0
        bad_points = 0
        for j in range(totalClusters):
            if j != i:
                distance = gpxpy.geo.haversine_distance(latitude_1 = regionCenters[i][0], longitude_1 = regionCenters[i][1], latitude_2 = regionCenters[j][0], longitude_2 = regionCenters[j][1])
                distance = distance/(1.60934*1000)   #distance from meters to miles
                min_distance = min(min_distance, distance) #it will return minimum of "min_distance, distance".
                if distance < 2:
                    good_points += 1
                else:
                    bad_points += 1
        less_dist.append(good_points)
        more_dist.append(bad_points)
    print("On choosing a cluster size of {}".format(totalClusters))
    print("Avg. Number clusters within vicinity where inter cluster distance < 2 miles is {}".format(np.ceil(sum(less_dist)/len(less_dist))))
    print("Avg. Number clusters outside of vicinity where inter cluster distance > 2 miles is {}".format(np.ceil(sum(more_dist)/len(more_dist))))
    print("Minimum distance between any two clusters = {}".format(min_distance))
    print("-"*10)
            
def makingRegions(noOfRegions):
    regions = MiniBatchKMeans(n_clusters = noOfRegions, batch_size = 10000, random_state = 5).fit(coord)
    regionCenters = regions.cluster_centers_ 
    totalClusters = len(regionCenters)
    return regionCenters, totalClusters


startTime = datetime.now()
for i in range(10, 100, 10):
    regionCenters, totalClusters = makingRegions(i)
    min_distance(regionCenters, totalClusters)
print("Time taken = "+str(datetime.now() - startTime))

coord = df[["pick_lat", "pick_lng"]].values
regions = MiniBatchKMeans(n_clusters = 50, batch_size = 10000, random_state = 0).fit(coord)
df["pickup_cluster"] = regions.predict(df[["pick_lat", "pick_lng"]])

df


### Model to Define pickup cluster, given latitude and longitude
dump(regions, 'D:\\Project_Pro\\ML_Path_Moderate\\Project_10\\pickup_cluster_model.joblib', compress = 3)

### These pickup clusters tell in which area most ride requests are coming. 
### Plotting Regions in Bangalore (our most rides requests here)
#### Bangalore:'boundingbox': ['12.8340125', '13.1436649', '77.4601025', '77.7840515']
bangalore_latitude_range = (12.8340125, 13.1436649)
bangalore_longitude_range = (77.4601025, 77.7840515)
fig = plt.figure()
ax = fig.add_axes([0,0,1.5,1.5])
ax.scatter(x = df.pick_lng.values[:100000], y = df.pick_lat.values[:100000], c = df.pickup_cluster.values[:100000], cmap = "Paired", s = 5)
ax.set_xlim(77.4601025, 77.7840515)
ax.set_ylim(12.8340125, 13.1436649)
ax.set_title("Regions in Bangalore")
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()
#Longitude values vary from left to right i.e., horizontally
#Latitude values vary from top to bottom means i.e., vertically

### Summing Ride Request count to 30mins interval groupby pickup cluster.

def round_timestamp_30interval(x):
    if type(x)==str:
        x = datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    return x- timedelta(minutes=x.minute%30, seconds=x.second, microseconds=x.microsecond)

df['ts'] = np.vectorize(round_timestamp_30interval)(df['ts'])

dataset = deepcopy(df)
dataset.ts = pd.to_datetime(dataset.ts)
dataset

dataset = dataset[['ts','number','pickup_cluster']]

dataset=dataset.groupby(by = ['ts','pickup_cluster']).count().reset_index()
dataset.columns = ['ts','pickup_cluster','request_count']
dataset

## Adding Dummy pickup cluster -1

## Change this Data based on your data
l = [datetime(2020,3,26,00,00,00) + timedelta(minutes = 30*i) for i in range(0,48*365)]
lt = []
for x in l:
    lt.append([x, -1, 0])
temp = pd.DataFrame(lt, columns = ['ts','pickup_cluster','request_count'])
dataset = dataset.append(temp,ignore_index=True)

data = dataset.set_index(['ts', 'pickup_cluster']).unstack().fillna(value=0).asfreq(freq='30Min').stack().sort_index(level=1).reset_index()

# Removing Dummy Cluster
data = data[data.pickup_cluster>=0]

assert len(data)==878400

### Adding Time features (hours, mins ,dayofweek, qurter and month)

data['mins'] = data.ts.dt.minute
data['hour'] = data.ts.dt.hour
data['day'] = data.ts.dt.day
data['month'] = data.ts.dt.month
data['dayofweek'] = data.ts.dt.dayofweek
data['quarter'] = data.ts.dt.quarter

data.to_csv('D:\\Project_Pro\\ML_Path_Moderate\\Project_10\\Data1_Prepared.csv',index = False, compression = 'gzip')

profile = data.profile_report(title='Ride Request DataSet Analysis')
profile.to_file(output_file="data_analysis_ride_request.html")

""" 
We now have clean, good ride requests data. Cluster of Latitude-Longitude is done, we have around 50 pickup_clusters.
We have grouped ride request day in 30mins interal. Total Data Rows: 366days*48 intervals * 50 clusters = 878400
AIM : To forecast demand for a given lattitude-longitude
Metric: RMSE, how close we are able to predict ride demand to true value.
"""

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from math import sqrt, ceil, floor
import matplotlib.pyplot as plt
from xgboost import plot_importance
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from joblib import dump, load

df = pd.read_csv('D:\\Project_Pro\\ML_Path_Moderate\\Project_10\\Data_prepared.csv', compression = 'gzip')
df.info()

df['request_count'] = pd.to_numeric(df['request_count'], downcast = 'integer')
df.ts = pd.to_datetime(df.ts)
df.head(10)

df = df[['ts','pickup_cluster','mins','hour','month','quarter','dayofweek','request_count']]
# First 24days of every month in train
df_train = df[df.ts.dt.day <= 23]
# Last 7 days of every month in test
df_test = df[df.ts.dt.day > 23]
len(df_train), len(df_test)

X = df_train.iloc[:, 1:-1]
y = df_train.iloc[:, -1]
X_test = df_test.iloc[:, 1:-1]
y_test = df_test.iloc[:,-1]

def metrics_calculate(regressor):
    y_pred = regressor.predict(X_test)
    rms = sqrt(mean_squared_error(y_test, y_pred))
    return rms

## Iteration : 1
# Features: ['pickup_cluster','mins','hour','month','quarter','dayofweek']

from sklearn.linear_model import LinearRegression
regressor = LinearRegression().fit(X, y)
print("RMSE TRAIN: {}, RMSE TEST: {}".format(sqrt(mean_squared_error(y, regressor.predict(X))), metrics_calculate(regressor)))

#### High Bais | Underfit 
# It is used to determine the extent to which there is a linear relationship between a dependent variable and one or more independent variables.
# Moving to Ensemble Regressors: Bagging Algorithm Random Forest
# Random Forest Regressor: Random Forests are bagged decision tree models that split on a subset of features on each split.

# Give Feature importance base on target variable.

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state=42, n_jobs = -1, verbose=True)
regressor.fit(X,y)
print("RMSE TRAIN: {}, RMSE TEST:{}".format(sqrt(mean_squared_error(y, regressor.predict(X))), metrics_calculate(regressor)))

feature_importances = pd.DataFrame(regressor.feature_importances_,
                                   index = X.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances

# Random Forest tend to Overfit

### Moving to a Ensemble: Boosting Algorithm

## XGBoost
import xgboost as xgb
model=xgb.XGBRegressor(learning_rate=0.01, random_state=0, n_estimators=1000, max_depth=8, objective="reg:squarederror")

eval_set = [(X_test, y_test)]
model.fit(X,y,verbose=True, eval_set=eval_set, early_stopping_rounds=15,eval_metric="rmse")
print("XGBOOST Regressor")
print("Model Score:",model.score(X,y))
print("RMSE TRAIN: {}, RMSE TEST:{}".format(sqrt(mean_squared_error(y, model.predict(X))), metrics_calculate(model)))

dump(model,'D:\\Project_Pro\\ML_Path_Moderate\\Project_10\\prediction_model_without_lag.joblib',compress=3)

# Plot feature imotance
plot_importance(model)

##### Iteration 2
# Include : Lag Features
# If there is 30min trend, how has ride requests been in last 1.5 hours.

plot_acf(df_train['request_count'], lags = 10)
plot_pacf(df_train['request_count'], lags = 10)

df_test = df_test.sort_values(by=['pickup_cluster', 'ts']).drop_duplicates(subset=['ts','pickup_cluster'])
temp = pd.concat([df_train,df_test])
temp = temp.sort_values(by=['pickup_cluster', 'ts']).drop_duplicates(subset=['ts','pickup_cluster'])
temp = temp.set_index(['ts', 'pickup_cluster', 'mins','hour', 'month', 'quarter', 'dayofweek'])

temp['lag_1'] = temp.groupby(level=['pickup_cluster'])['request_count'].shift(1)
temp['lag_2'] = temp.groupby(level=['pickup_cluster'])['request_count'].shift(2)
temp['lag_3'] = temp.groupby(level=['pickup_cluster'])['request_count'].shift(3)

temp = temp.reset_index(drop = False).dropna()
temp = temp[['ts', 'pickup_cluster', 'mins','hour', 'month', 'quarter',
       'dayofweek', 'lag_1', 'lag_2', 'lag_3',
        'request_count']]


train1 = temp[temp.ts.dt.day <=23]
test1 = temp[temp.ts.dt.day >23]

X = train1.iloc[:, 1:-1]
y = train1.iloc[:, -1]
X_test = test1.iloc[:, 1:-1]
y_test = test1.iloc[:, -1]

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 500, random_state=0, n_jobs = -1)
regressor.fit(X,y)
print("RMSE TRAIN: {}, RMSE TEST:{}".format(sqrt(mean_squared_error(y, regressor.predict(X))), metrics_calculate(regressor)))

# Random forest overfits; moving to Boosting Algorithm

feature_importances = pd.DataFrame(regressor.feature_importances_,
                                   index = X.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances



## XGBoost
import xgboost as xgb
model=xgb.XGBRegressor(learning_rate=0.01, random_state=0, n_estimators=600, max_depth=8, objective="reg:squarederror")

eval_set = [(X_test, y_test)]
model.fit(X,y,verbose=True, eval_set=eval_set, early_stopping_rounds=30,eval_metric="rmse")
print("XGBOOST Regressor")
print("Model Score:",model.score(X,y))
print("RMSE TRAIN: {}, RMSE TEST:{}".format(sqrt(mean_squared_error(y, model.predict(X))), metrics_calculate(model)))

plot_importance(model)

## Xgboost performance better than random forest.


##### Iteration 3
# Include: Both Lag Features and Rolling Window
# Both of size = 3


df_test = df_test.sort_values(by=['pickup_cluster', 'ts']).drop_duplicates(subset=['ts','pickup_cluster'])
temp = pd.concat([df_train,df_test])
temp = temp.sort_values(by=['pickup_cluster', 'ts']).drop_duplicates(subset=['ts','pickup_cluster'])
temp = temp.set_index(['ts', 'pickup_cluster', 'mins', 'hour', 'month', 'quarter', 'dayofweek'])

temp['lag_1'] = temp.groupby(level=['pickup_cluster'])['request_count'].shift(1)
temp['lag_2'] = temp.groupby(level=['pickup_cluster'])['request_count'].shift(2)
temp['lag_3'] = temp.groupby(level=['pickup_cluster'])['request_count'].shift(3)
temp['rolling_mean'] = temp.groupby(level=['pickup_cluster'])['request_count'].apply(lambda x: x.rolling(window = 3).mean()).shift(1)

temp = temp.reset_index(drop = False).dropna()
temp = temp[['ts', 'pickup_cluster', 'mins', 'hour', 'month', 'quarter',
       'dayofweek', 'lag_1', 'lag_2', 'lag_3','rolling_mean','request_count']]
train1 = temp[temp.ts.dt.day <=23]
test1 = temp[temp.ts.dt.day >23]

X = train1.iloc[:, 1:-1]
y = train1.iloc[:, -1]
X_test = test1.iloc[:, 1:-1]
y_test = test1.iloc[:, -1]

# from sklearn.ensemble import RandomForestRegressor
# regressor = RandomForestRegressor(n_estimators = 500, random_state=0, n_jobs = -1)
# regressor.fit(X,y)
# print("RMSE TRAIN: {}, RMSE TEST:{}".format(sqrt(mean_squared_error(y, regressor.predict(X))), metrics_calculate(regressor)))

# feature_importances = pd.DataFrame(regressor.feature_importances_,
#                                    index = X.columns,
#                                     columns=['importance']).sort_values('importance',ascending=False)
# feature_importances

## Random Forest Overfits(performance good at Train and bad at Test)

## XGBoost
import xgboost as xgb
model=xgb.XGBRegressor(learning_rate=0.01, random_state=0, n_estimators=1500, max_depth=8, objective="reg:squarederror")

eval_set = [(X_test, y_test)]
model.fit(X,y,verbose=True, eval_set=eval_set, early_stopping_rounds=20,eval_metric="rmse")
print("XGBOOST Regressor")
print("Model Score:",model.score(X,y))
print("RMSE TRAIN: {}, RMSE TEST:{}".format(sqrt(mean_squared_error(y, model.predict(X))), metrics_calculate(model)))

plot_importance(model)

dump(model,'D:\\Project_Pro\\ML_Path_Moderate\\Project_10\\prediction_model.joblib',compress=3)

### Lag Features and Rolling Mean has improved RMSE
# Final model : iteration3
# Good performance with feature addition

""" Has Features: ['pickup_cluster','mins','hour','month','dayofweek',
'quarter', 'lag_1','lag_2','lag_3','rolling_mean']
    
    Model score: 0.91"""

model = load('D:\\Project_Pro\\ML_Path_Moderate\\Project_10\\prediction_model.joblib')

model.predict(X_test)


'''   
Task: Given 26th March 2021 demand-clean dataset based on business team rules; we need to forecast
demand for first 3(30 mins internal) demand value of 27 th March 2021. 

'''

import pandas as pd
import numpy as np
from joblib import load, dump
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt, ceil, floor
from datetime import datetime, timedelta

def round_timestamp_30interval(x):
    if type(x)==str:
        x = datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    return x- timedelta(minutes=x.minute%30, seconds=x.second, microseconds=x.microsecond)

def time_features(data):
    data['mins'] = data.ts.dt.minute
    data['hour'] = data.ts.dt.hour
    data['day'] = data.ts.dt.day
    data['month'] = data.ts.dt.month
    data['dayofweek'] = data.ts.dt.dayofweek
    data['quarter'] = data.ts.dt.quarter
    return data

def prediction_without_lag(df):
    return predict_without_lag.predict(df[['pickup_cluster','mins','hour','month','quarter','dayofweek']])

def prediction_with_lag(df):
    return predict_with_lag.predict(df[['pickup_cluster', 'mins', 'hour', 'month', 'quarter',
           'dayofweek', 'lag_1', 'lag_2', 'lag_3','rolling_mean']])

def shift_with_lag_and_rollingmean(df):
    df = df.sort_values(by=['pickup_cluster', 'ts']).drop_duplicates(subset=['ts','pickup_cluster'])
    df = df.set_index(['ts', 'pickup_cluster', 'mins', 'hour', 'month', 'quarter', 'dayofweek'])
    df['lag_1'] = df.groupby(level=['pickup_cluster'])['request_count'].shift(1)
    df['lag_2'] = df.groupby(level=['pickup_cluster'])['request_count'].shift(2)
    df['lag_3'] = df.groupby(level=['pickup_cluster'])['request_count'].shift(3)
    df['rolling_mean'] = df.groupby(level=['pickup_cluster'])['request_count'].apply(lambda x: x.rolling(window = 3).mean()).shift(1)

    df = df.reset_index(drop = False).dropna()
    df = df[['ts', 'pickup_cluster', 'mins', 'hour', 'month', 'quarter',
           'dayofweek', 'lag_1', 'lag_2', 'lag_3','rolling_mean','request_count']]
    return df


df = pd.read_csv('D:\\Project_Pro\\ML_Path_Moderate\\Project_10\\cleaned_test_booking_data.csv', compression = 'gzip', low_memory=False)
cluster_model = load('D:\\Project_Pro\\ML_Path_Moderate\\Project_10\\pickup_cluster_model.joblib')
predict_without_lag = load('D:\\Project_Pro\\ML_Path_Moderate\\Project_10\\prediction_model_without_lag.joblib')
predict_with_lag = load('D:\\Project_Pro\\ML_Path_Moderate\\Project_10\\prediction_model.joblib')

# Use Clustering Kmeans Model for Geospacial Feature - pickup_cluster

df['pickup_cluster'] = cluster_model.predict(df[['pick_lat','pick_lng']])

# Data preparation and processing
df['ts'] = np.vectorize(round_timestamp_30interval)(df['ts'])
df['ts'] = pd.to_datetime(df['ts'])

df = df[['ts','number','pickup_cluster']]
df=df.groupby(by = ['ts','pickup_cluster']).count().reset_index()
df.columns = ['ts','pickup_cluster','request_count']

## Adding Dummy pickup cluster -1

## Change this Data based on your data
l = [datetime(2021,3,26,00,00,00) + timedelta(minutes = 30*i) for i in range(0,51)]
lt = []
for x in l:
    lt.append([x, -1, 0])
temp = pd.DataFrame(lt, columns = ['ts','pickup_cluster','request_count'])
df = df.append(temp,ignore_index=True)

data = df.set_index(['ts', 'pickup_cluster']).unstack().fillna(value=0).asfreq(freq='30Min').stack().sort_index(level=1).reset_index()

# Removing Dummy Cluster
data = data[data.pickup_cluster>=0]

df = time_features(data)

# Model without Lag(past data) requirement.

data_without_lag = df[df['ts']>=datetime(2021,3,27,00,00,00)].__copy__()
data_without_lag['request_count'] = prediction_without_lag(data_without_lag)
data_without_lag

data_without_lag.to_csv('D:\\Project_Pro\\ML_Path_Moderate\\Project_10\\prediction_without_lag_model.csv',index = False, compression = 'gzip')

## Using iteration 3 - Best Model with Lag Features and Rolling Means(Recursive Multi-Step Forecast used)

start_date = datetime(2021,3,27,00,00,00) 
for x in range(3):
    df = shift_with_lag_and_rollingmean(df)
    df.loc[df[df['ts']==start_date+timedelta(minutes=30*x)].index,'request_count'] = prediction_with_lag(df[df['ts']==start_date+timedelta(minutes=30*x)])
    
data_with_lag = df[df['ts']>=datetime(2021,3,27,00,00,00)].__copy__()
data_with_lag


data_with_lag.to_csv('D:\\Project_Pro\\ML_Path_Moderate\\Project_10\\prediction_with_lag_model.csv',index = False, compression = 'gzip')





























































































