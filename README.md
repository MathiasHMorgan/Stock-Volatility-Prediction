# Stock Volatility Prediction

📌 Project Overview

This project focuses on predicting future stock market volatility using a combination of historical trading data and quarterly financial metrics. By leveraging advanced feature engineering—such as cyclical time encoding and volatility clustering captures (EMA/SMA)—this assessment compares the performance of three powerful regression models: LightGBM, CatBoost, and Random Forest.

The objective is to minimize the Root Mean Squared Error (RMSE) on a hold-out validation set that simulates real-world temporal forecasting.

This is a solid foundation for a portfolio-grade GitHub repository. Your notebook demonstrates strong feature engineering—particularly the use of cyclical encoding and exponential moving averages (EMAs)—which clearly paid off given the low RMSE scores.

📊 Dataset Description

The dataset contains 13,486 rows of historical stock data, including:

Price Action: Open, Close, High, Low, Return.

Trading Metrics: Volume, Amount, Avg_Price.

Financial Fundamentals: Revenue, Net Income, EPS, Total Assets/Liabilities, Cash Flows.

Target: Volatility (transformed to Volatility_Log for modeling).

🛠️ Feature Engineering Highlights

To improve model convergence and capture temporal dependencies, the following transformations were applied:

Log Transformation: Raw volatility was heavily right-skewed. np.log1p was used to normalize the distribution.

Cyclical Encoding: Months were transformed into Sine and Cosine components to preserve the geometric relationship between December and January.

Lagged Features: 1, 2, and 3-month lags were created for volatility and trading volume to capture volatility clustering.

Moving Averages: Simple (SMA) and Exponential (EMA) moving averages (3 and 6-period windows) were used to capture momentum and trend signals.
