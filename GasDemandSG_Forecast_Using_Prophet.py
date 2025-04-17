# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 13:25:49 2025

@author: Jun Hui
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

#import the datafile and understand the data
datafile = pd.read_csv("GasDemandSG.csv")
print(datafile.head())
print(datafile.info())

#Convert month column into datetime
datafile.month = pd.to_datetime(datafile.month)

#Filter out data of interest (4rm in this case)
four_room = datafile[datafile['dwelling_type'] == '4-Room']

#Visualise how the data points are
#Prophet model is known to handle outliers well.
#Outliers seen in 2020 is due to covid lockdowns in Singapore, which led to spike in resid gas demand
plt.scatter(four_room.month, four_room.tg_consumption_gwh)
plt.title("Gas Demand from 2005 to 2022")
plt.xlabel("Month")
plt.ylabel("Demand (gwh)")

#Running the forecast with covid datapoints
#Setting up the variables
four_room['ds'] = datafile['month']
four_room['y'] = four_room['tg_consumption_gwh']

df= four_room[['ds','y']]

#importing ML model Prophet
from prophet import Prophet

# Initialize and fit the model
model = Prophet(growth='linear', yearly_seasonality=True)
model.fit(df)

# Create a dataframe for future dates (e.g., next 12 months)
future = model.make_future_dataframe(periods=12, freq='ME')

# Forecast
forecast = model.predict(future)

# Show forecasted values
#print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))

# Plot the forecast
fig1 = model.plot(forecast)
plt.title('Monthly Demand Forecast - Before IQE Outlier Removal')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.show()

# Extract actual and predicted values for comparable periods
comparison_df = pd.merge(
    df[['ds', 'y']], 
    forecast[['ds', 'yhat']], 
    on='ds', 
    how='inner'
)

actual = comparison_df['y'].values
predicted = comparison_df['yhat'].values

# Calculate metrics
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))

# Calculate MAPE
mape = np.mean(np.abs((actual - predicted) / actual)) * 100

# Calculate WMAPE
wmape = np.sum(np.abs(actual - predicted)) / np.sum(actual) * 100

# Calculate Bias
bias = np.sum(predicted - actual) / np.sum(actual) * 100

print("Forecast Accuracy Metrics for Dataset with Outliers ")
print(f"MAE: {mae:.2f} gwh")
print(f"RMSE: {rmse:.2f} gwh")
print(f"MAPE: {mape:.2f}%")
print(f"WMAPE: {wmape:.2f}%")
print(f"Bias: {bias:.2f}%")




### Running the forecast without covid datapoints
## We will be using IQR to remove extreme outliers
# Calculate IQR
Q1 = four_room['tg_consumption_gwh'].quantile(0.25)
Q3 = four_room['tg_consumption_gwh'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
four_room_filtered_iqr = four_room[(four_room['tg_consumption_gwh'] >= lower_bound) & (four_room['tg_consumption_gwh'] <= upper_bound)]

# Visualize before and after
plt.scatter(four_room_filtered_iqr.month, four_room_filtered_iqr.tg_consumption_gwh)
plt.title("Gas Demand from 2005 to 2022 - After IQR Outlier Removal")
plt.xlabel("Month")
plt.ylabel("Demand (gwh)")

#Setting up the variables
four_room_filtered_iqr['ds'] = four_room_filtered_iqr['month']
four_room_filtered_iqr['y'] = four_room_filtered_iqr['tg_consumption_gwh']

df_filter= four_room_filtered_iqr[['ds','y']]

# Initialize and fit the model
model = Prophet(growth='linear', yearly_seasonality=True)
model.fit(df_filter)

# Create a dataframe for future dates (e.g., next 12 months)
future_b = model.make_future_dataframe(periods=12, freq='ME')

# Forecast
forecast_b = model.predict(future_b)

# Show forecasted values for dataset without outliers
#print(forecast_b[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))

#Plot the forecast
fig1 = model.plot(forecast_b)
plt.title('Monthly Demand Forecast - After IQE Outlier Removal')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.show()

# Extract actual and predicted values for comparable periods
comparison_df = pd.merge(
    df_filter[['ds', 'y']], 
    forecast_b[['ds', 'yhat']], 
    on='ds', 
    how='inner'
)

actual_b = comparison_df['y'].values
predicted_b = comparison_df['yhat'].values

# Calculate metrics
mae = mean_absolute_error(actual_b, predicted_b)
rmse = np.sqrt(mean_squared_error(actual_b, predicted_b))

# Calculate MAPE
mape = np.mean(np.abs((actual_b - predicted_b) / actual_b)) * 100

# Calculate WMAPE
wmape = np.sum(np.abs(actual_b - predicted_b)) / np.sum(actual_b) * 100

# Calculate Bias
bias = np.sum(predicted_b - actual_b) / np.sum(actual_b) * 100

print("Forecast Accuracy Metrics for Dataset without Outliers ")
print(f"MAE: {mae:.2f} gwh")
print(f"RMSE: {rmse:.2f} gwh")
print(f"MAPE: {mape:.2f}%")
print(f"WMAPE: {wmape:.2f}%")
print(f"Bias: {bias:.2f}%")
