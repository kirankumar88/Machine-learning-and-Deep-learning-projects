USD–AUD Time Series Forecasting using ETS and ARIMA
📌 Project Overview

This repository contains a comparative time-series forecasting analysis of the USD–AUD exchange rate using classical statistical models. The project evaluates Double Exponential Smoothing (ETS) and ARIMA in modeling trend dynamics and short-term dependencies in foreign exchange data.
The emphasis is on statistical correctness, transparent evaluation, and realistic interpretation, acknowledging the stochastic nature of financial markets.

🎯 Objectives
Explore and visualize USD–AUD exchange rate data
Perform time-series decomposition (trend, seasonality, residuals)
Build ETS and ARIMA forecasting models
Evaluate models using standard error metrics
Compare model performance and generate future forecasts
Derive actionable analytical insights and limitations

📂 Repository Contents (Single-Folder Structure)
All files are intentionally maintained in a single directory for simplicity.
usd-aud-time-series-forecasting/
│
├── Exchange_rate.csv
├── usd_aud_time_series_forecasting.ipynb
├── exchange_rate_plot.jpeg
├── Residual_plot.jpeg
├── train_test_plot.jpeg
├── DES_predicted_plot.jpeg
├── DES_future_forecast_plot.jpeg
├── ACF_plot.jpeg
├── PACF_plot.jpeg
├── Plot_diagnostics.jpeg
├── ARIMA_test_forecast_plot.jpeg
├── ARIMA_forecast_plot.jpeg
├── README.md
└── requirements.txt
📊 Dataset

Description: Historical USD–AUD daily exchange rate data
Records: 7,588 observations
Frequency: Daily
Missing values: None
Target variable: Ex_rate

🔍 Methodology
1. Exploratory Data Analysis
Trend and volatility inspection
Missing and zero-value validation
Stationarity assessment

2. Time-Series Decomposition
STL decomposition (robust mode)
Findings:
Dominant trend component
Negligible seasonality
Volatility-driven residuals

3. Forecasting Models
🔹 Double Exponential Smoothing (ETS)

Additive trend
No seasonal component
Used as a baseline trend model

🔹 ARIMA
Stationarity checked using ADF test
First-order differencing applied

Model order selected via auto_arima

Final model: ARIMA(3,1,0)

📈 Model Evaluation
Metrics Used

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

Mean Absolute Percentage Error (MAPE)

Performance Comparison
Model	MAE	RMSE	MAPE (%)
ETS (DES)	~0.134	~0.151	~17.9
ARIMA(3,1,0)	~0.125	~0.140	~16.6

Outcome: ARIMA consistently outperforms ETS, though gains are modest.

🔮 Forecasting

Out-of-sample test forecasting for model validation
Three-year future forecast using the full dataset
Forecasts exhibit mean-reverting behavior, consistent with FX market theory

💡 Key Insights
USD–AUD exchange rates show limited predictability
Long-term directional forecasting is unreliable using price history alone
ARIMA captures short-term dependencies better than ETS
Volatility and external shocks dominate exchange-rate movements
Classical models provide analytical baselines, not trading strategies

Limitations :
No exogenous macroeconomic variables included
Volatility models (e.g., GARCH) not applied
Forecast confidence intervals not explicitly modeled
Not intended for financial trading decisions

🛠️ Tech Stack
Python
pandas, numpy
matplotlib, seaborn
statsmodels
pmdarima
scikit-learn

🚀 How to Run
pip install -r requirements.txt

Open the notebook:

jupyter notebook usd_aud_time_series_forecasting.ipynb

📌 Conclusion
This project demonstrates that while classical time-series models effectively capture structural properties of financial data, their predictive power remains limited in highly volatile markets. ARIMA provides marginal improvements over ETS, reinforcing the importance of realistic expectations in financial forecasting.
