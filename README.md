**Abstract**

This study evaluates Facebook’s Prophet algorithm for forecasting monthly gas consumption in Singapore’s 4-room public housing (HDB) sector. This analysis demonstrates that retaining COVID-19-induced outliers in the dataset yields superior real-world forecasting performance compared to interquartile range (IQR) filtering. While both approaches achieved high accuracy (MAPE < 7%), the unfiltered model exhibited better generalization on test data containing pandemic-era anomalies. The research validates Prophet’s inherent robustness to extreme events while maintaining exceptional baseline accuracy, providing critical insights for utility planning under uncertain conditions.

**Introduction**

Singapore’s residential gas demand forecasting faces unique challenges from urbanization trends and unexpected disruptions like COVID-19 lockdowns. This study analyzes 2005-2022 consumption data for 4-room HDB flats.
The primary objectives are:
1. Compare Prophet’s performance on raw vs outlier-filtered data
2. Quantify model robustness to pandemic-scale anomalies
3. Produce actionable 2030 demand projections for infrastructure planning

**Methodology**

_Data Handling

The raw time series data exhibited clear upward trends and seasonal patterns with notable outliers during 2020-2021, coinciding with Singapore's COVID-19 lockdown periods. 
Chronological 80/20 train-test split preserving temporal structure
Dual pipeline implementation:
- Unfiltered: Retains all data including COVID-19 outliers
- IQR-filtered: Removes training set values beyond [25% and 75%]

_Results_

![image](https://github.com/user-attachments/assets/244efbc4-cf71-4b05-b3b7-dae32411b98f)

Key Takeaways:
1. Outlier Resilience: Prophet achieved 6.09% MAPE on unfiltered data despite COVID-19 anomalies, outperforming the filtered model. That said, both models outperformed industry standards for demand forecasting of ~10% to 20%
2. Error Consistency: Both MAE and RMSE increased when using the filtered model, indicating that the filtered model is relatively less accurate.
3. Biasness: Bias increased negatively as we used the filtered model. 

**Why Retention Outperforms Filtering**
1. Prophet’s Anomaly Handling: The multiplicative seasonality configuration automatically downweights extreme values through uncertainty interval expansion.
2. Information Preservation: COVID-19 patterns contained valid demand signals about emergency response behaviors.
3. Future-Proofing: Retained anomalies improve model readiness for similar disruptions.

**Conclusion and Future Work**
This study establishes that Prophet delivers superior gas demand forecasts for Singapore’s HDB sector when retaining pandemic-era outliers, achieving:
- Operational Accuracy: 6.09% MAPE on real-world test data
- Strategic Value: Actionable 7-year projections with uncertainty bounds

The step course of action would be to incorporate COVID-like events as explicit regressors. This would allow the model to make more decisive predictions.  

**Charts from this analysis**

Raw Dataset:
![image](https://github.com/user-attachments/assets/d5723f65-d41b-4385-8edf-88a4f01f6b18)

Prophet Forecast with 80/20 Train/Test Split
![image](https://github.com/user-attachments/assets/6882b9ee-402c-43c5-8d09-87a0e0d835a8)

Prophet Forecast with 80/20 Train/Test Split with IQR Applied on Train Set
![image](https://github.com/user-attachments/assets/d271fd30-fa58-4e20-a00c-4beb331eac15)

Prophet Forecast for 4-Room HDB Gas Demand till 2030
![image](https://github.com/user-attachments/assets/24e50eeb-e54f-4a07-ab75-80759b4d89b5)


