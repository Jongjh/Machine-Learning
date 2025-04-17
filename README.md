**Abstract**

This study evaluates the efficacy of Facebook's Prophet algorithm for forecasting monthly gas consumption in Singapore's public housing sector, specifically 4-room Housing Development Board (HDB) flats. 
The research implements a comparative analysis between models trained on complete historical data versus a dataset with pandemic-related outliers removed. 
Performance metrics demonstrate that while both approaches yield high accuracy, outlier removal significantly enhances forecast precision. 
The findings validate Prophet's robustness for time series forecasting in utilities consumption modeling, particularly when seasonal patterns and anomalous events are present.

**Introduction**

This research leverages publicly available time series data on Singapore's residential gas consumption, focusing specifically on 4-room public housing units (HDBs). 
The longitudinal dataset spans from 2005 to 2022, capturing the monthly gas demand patterns across varying economic and social conditions, including the COVID-19 pandemic period. 
The primary objective is to develop and evaluate a machine learning forecasting model capable of effectively capturing seasonal patterns while producing reliable demand projections for utility planning.

**Methodology**

_Data Preprocessing_

The raw time series data exhibited clear upward trends and seasonal patterns with notable outliers during 2020-2021, coinciding with Singapore's COVID-19 lockdown periods. 
These anomalous consumption patterns—reaching approximately 28-32 GWh compared to the typical range of 15-25 GWh—provided an opportunity to evaluate model performance under exceptional circumstances.

_Outlier Treatment_

Employing interquartile range (IQR) methodology, outliers were systematically identified and removed from a parallel dataset to create two distinct training scenarios:
1. Complete dataset including pandemic-period consumption anomalies
2. Filtered dataset with outliers removed

_Model Implementation_

Facebook's Prophet algorithm was selected for its demonstrated effectiveness in handling time series with:
1. Strong seasonal components
2. Missing data points
3. Trend changes
4. Outlier resilience

The model was configured with linear growth parameters and yearly seasonality components, then trained separately on both the complete and filtered datasets.

**Results**

Comprehensive evaluation metrics were calculated to assess forecast accuracy:

![image](https://github.com/user-attachments/assets/31280e85-52aa-4b1d-b809-e9c91c7c2a8e)


**Statistical Analysis**

Both models demonstrated excellent predictive performance with MAPE values under 3%, significantly outperforming industry standards for demand forecasting (typically 10-20%). 
However, several key observations emerged:

- The notable divergence between MAE (0.55 GWh) and RMSE (1.08 GWh) in the complete dataset confirms the significant impact of outliers on model performance, as RMSE disproportionately penalizes large errors.
- The filtered dataset exhibited minimal difference between MAE (0.34 GWh) and RMSE (0.46 GWh), indicating a more consistent error distribution without the influence of extreme values.
- WMAPE analysis revealed that the model's performance degradation was more pronounced during high-demand periods when outliers were present (3.13% vs. 1.93%).
- Both models maintained zero bias, demonstrating balanced error distribution without systematic over- or under-forecasting tendencies.

**Discussion**

The Prophet algorithm demonstrated exceptional forecasting capability for Singapore's residential gas consumption, with or without outlier treatment. 
The model effectively captured both the long-term upward trend and seasonal fluctuations visible in the time series.
When outliers were removed, the model achieved outstanding accuracy (MAPE: 1.88%), which represents a 35.2% improvement over the model trained on the complete dataset. 
This substantial enhancement underscores the value of proper outlier management in time series forecasting applications, particularly when anomalous events (such as pandemic lockdowns) create temporary but significant deviations from established patterns.
The negligible difference between MAE and RMSE in the filtered dataset confirms successful outlier elimination, resulting in more consistent and reliable predictions across the entire range of the forecast horizon.

**Conclusion and Implications**
This study demonstrates that Prophet's machine learning approach produces highly accurate forecasts for monthly gas consumption in Singapore's residential sector, with error metrics well below industry standards. 
The research quantifies the specific benefits of outlier removal in improving forecast accuracy, providing a methodological framework for demand forecasting in utility planning.
The findings have significant implications for energy resource allocation, infrastructure planning, and policy development. 
Future research could extend this approach to other housing types or utilities, incorporate additional exogenous variables such as temperature or economic indicators, or explore ensemble methods to further enhance predictive performance.
