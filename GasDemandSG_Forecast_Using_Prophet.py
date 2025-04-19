# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 13:25:49 2025

@author: Jun Hui
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np

#import the datafile and understand the data
datafile = pd.read_csv("C:/Users/Jun Hui/Desktop/Studying Materials/GasDemandSG.csv")
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

###Running the forecast with covid datapoints

#Setting up the variables
four_room['ds'] = four_room['month']
four_room['y'] = four_room['tg_consumption_gwh']
df= four_room[['ds','y']]

#splitting data into 80% train and 20% test
#set shuffle = False to avoid random splitting of timeseries which will eliminate seasonality factor
train, test = train_test_split(four_room, test_size=0.2, shuffle=False)

#importing ML model Prophet
from prophet import Prophet

# Initialize and fit the model
model = Prophet(growth='linear', yearly_seasonality=True, seasonality_mode='multiplicative',changepoint_prior_scale=0.05)
model.fit(train)

# Create a dataframe for future dates in test df.
future = model.make_future_dataframe(periods=len(test), freq='MS')

# Forecast for the test period
forecast = model.predict(future)

#Retain predictions for the test period
forecast_test = forecast.tail(len(test)).reset_index(drop=True)
test = test.reset_index(drop = True)

#Merge actual and predicted for test period
comparison_df = pd.concat([test, forecast_test['yhat']],axis=1)

#Calculate metrics on test results
actual = comparison_df['y'].values
predicted = comparison_df['yhat'].values
mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)
mape = (np.abs((actual - predicted) / actual)).mean() * 100
wmape = (np.abs(actual - predicted).sum() / actual.sum()) * 100
bias = ((predicted - actual).sum() / actual.sum()) * 100


# Plot the test results
plt.figure(figsize=(12, 6))
plt.plot(train['ds'], train['y'], label='Train Data')
plt.plot(test['ds'], test['y'], label='Test Data')
plt.plot(test['ds'], predicted, label='Prophet Forecast (Test)')
plt.title('Prophet Forecast with 80/20 Train/Test Split')
plt.xlabel('Date')
plt.ylabel('Gas Demand (GWh)')

plt.fill_between(forecast_test['ds'], 
                forecast_test['yhat_lower'], 
                forecast_test['yhat_upper'],
                color='blue', alpha=0.1, label='95% CI')

plt.legend()
plt.show()

# Print metrics
print("Forecast Accuracy Metrics on Test Set")
print(f"MAE: {mae:.2f} gwh")
print(f"RMSE: {rmse:.2f} gwh")
print(f"MAPE: {mape:.2f}%")
print(f"WMAPE: {wmape:.2f}%")
print(f"Bias: {bias:.2f}%")

### Check if there will be a improvement in metrics by applying IQR on train df
train, test = train_test_split(four_room, test_size=0.2, shuffle=False)
Q1 = train['tg_consumption_gwh'].quantile(0.25)
Q3 = train['tg_consumption_gwh'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Apply to both train and test
train_filtered = train[(train['tg_consumption_gwh'] >= lower_bound) & 
                      (train['tg_consumption_gwh'] <= upper_bound)]


# Initialize and fit the model
model_filtered = Prophet(growth='linear', yearly_seasonality=True)
model_filtered.fit(train_filtered)

# Create a dataframe for future dates in test df.
future_filtered = model_filtered.make_future_dataframe(periods=len(test), freq='ME')

# Forecast for the test period
forecast_filtered = model_filtered.predict(future_filtered)

#Retain predictions for the test period
forecast_filtered_test = forecast_filtered.tail(len(test)).reset_index(drop=True)
test = test.reset_index(drop = True)

#Merge actual and predicted for test period
comparison_filtered_df = pd.concat([test, forecast_filtered_test['yhat']],axis=1)

#Calculate metrics on test results
actual_filtered = comparison_filtered_df['y'].values
predicted_filtered = comparison_filtered_df['yhat'].values
mae = mean_absolute_error(actual_filtered, predicted_filtered)
mse = mean_squared_error(actual_filtered, predicted_filtered)
rmse = np.sqrt(mse)
mape = (np.abs((actual_filtered - predicted_filtered) / actual_filtered)).mean() * 100
wmape = (np.abs(actual_filtered - predicted_filtered).sum() / actual_filtered.sum()) * 100
bias = ((predicted_filtered - actual_filtered).sum() / actual_filtered.sum()) * 100


# Plot the test results
plt.figure(figsize=(12, 6))
plt.plot(train_filtered['ds'], train_filtered['y'], label='Train Data')
plt.plot(test['ds'], test['y'], label='Test Data')
plt.plot(test['ds'], predicted_filtered, label='Prophet Forecast (Test)')
plt.title('Prophet Forecast with  80/20 Train/Test Split with IQR applied')
plt.xlabel('Date')
plt.ylabel('Gas Demand (GWh)')

plt.fill_between(forecast_test['ds'], 
                forecast_test['yhat_lower'], 
                forecast_test['yhat_upper'],
                color='blue', alpha=0.1, label='95% CI')

plt.legend()
plt.show()

# Print metrics
print("Forecast Accuracy Metrics on Test Set with Outliers Removed")
print(f"MAE: {mae:.2f} gwh")
print(f"RMSE: {rmse:.2f} gwh")
print(f"MAPE: {mape:.2f}%")
print(f"WMAPE: {wmape:.2f}%")
print(f"Bias: {bias:.2f}%")


#conclusion: Metrics did not improve when train df is being filtered. As such, we can forcast 2030 demand using the original dataset - despite outlier in early 2020 due to covid. 

'''Forecasting demand till 2030'''

# Initialize and fit model with improved configuration
# 95% confidence interval
final_model = Prophet(growth='linear', yearly_seasonality=True, seasonality_mode='multiplicative', interval_width=0.95)
final_model.fit(df)

# Calculate number of months needed to forecast until end of 2030
last_date = df['ds'].max()
end_date = pd.to_datetime('2030-12-31')
forecast_months = (end_date.year - last_date.year) * 12 + (end_date.month - last_date.month)

# Create future dataframe
future_c = final_model.make_future_dataframe(periods=forecast_months, freq='MS',include_history=True)

# Generate forecast
forecast_c = final_model.predict(future_c)

fig = final_model.plot(forecast_c)
ax = fig.gca()
ax.set_ylim(0, 40)  # Adjust based on historical range
plt.fill_between(forecast_c['ds'], 
                forecast_c['yhat_lower'], 
                forecast_c['yhat_upper'],
                color='blue', alpha=0.1)
plt.title('4-Room HDB Gas Demand Forecast 2005-2030')
plt.xlabel('Date')
plt.ylabel('Demand')

plt.show()
