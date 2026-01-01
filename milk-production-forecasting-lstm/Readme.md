# Milk Production Forecasting with Deep Learning (RNN, LSTM, GRU)

## Overview
This project implements an end-to-end **time-series forecasting framework** to predict monthly milk production using **deep learning models**, including Simple RNN, GRU, and LSTM. The objective is to evaluate model performance in capturing **trend, seasonality, and long-term temporal dependencies**, and to translate predictive accuracy into **actionable business insights** for operational planning.

---

## Business Problem
Accurate milk production forecasts are essential for production scheduling, inventory optimization, logistics planning, and financial forecasting. Conventional models often struggle with long-term seasonality and non-linear growth patterns. This study assesses whether recurrent neural networks can provide **more stable and reliable forecasts** for real-world decision-making.

---

## Dataset
- Historical monthly milk production data  
- Strong annual seasonality (12-month cycle)  
- Long-term upward trend  
- No missing or zero values  

The dataset is well-suited for time-series modeling and forecasting applications.

---

## Methodology
The analytical workflow follows a structured pipeline:

1. Exploratory Data Analysis (EDA) to assess data quality, trends, and temporal structure  
2. STL decomposition to isolate trend, seasonal, and residual components  
3. Data preprocessing using Min-Max scaling and sliding-window sequence generation (window size = 12)  
4. Model development using Simple RNN, GRU, and LSTM architectures  
5. Model evaluation using RMSE, MAE, and MAPE metrics  
6. Twelve-month forward forecasting using the best-performing model  

Early stopping and validation splits are applied to ensure generalization and prevent overfitting.

---

## Results
Among all evaluated models, **LSTM demonstrated superior performance**:

- Lowest RMSE and MAE across test data  
- Mean Absolute Percentage Error (MAPE) consistently below 3%  
- Stable convergence with minimal overfitting  
- Strong capability to capture long-term seasonality and trend  

These results confirm LSTM as the most reliable architecture for monthly milk production forecasting.

---

## Business Impact
- High-confidence monthly production forecasts  
- Improved inventory and cold-chain planning  
- Reduced risk of overproduction and stock-outs  
- Optimized workforce and maintenance scheduling  
- Data-driven support for budgeting and revenue planning  

The 12-month forecast maintains realistic production ranges and seasonal patterns, making it suitable for operational use.

---

