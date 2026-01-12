# Stock Volatility Prediction: Time-Series Regression Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)
![Gradient Boosting](https://img.shields.io/badge/Models-CatBoost%20%7C%20LightGBM%20%7C%20RF-green.svg)

## 📌 Project Overview
This project predicts future stock market volatility by integrating historical price action with quarterly financial fundamentals. The primary challenge addressed is the temporal dependency of market data (volatility clustering). By implementing advanced feature engineering and applying the CatBoost model, this project identifies the most effective predictors of market risk.

## 📊 Dataset Overview
The dataset consists of **13,486 observations** including:
* **Market Data:** Open, Close, High, Low, Volume, and Return.
* **Fundamental Data:** Revenue, Net Income, EPS, Total Assets, Liabilities, and Cash Flow metrics.
* **Target Variable:** `Volatility` (Standard Deviation of returns).

## 🛠️ Feature Engineering & Preprocessing
To improve model performance and ensure statistical validity, several key transformations were performed:

* **Log Transformation:** Applied `np.log1p` to the target variable to mitigate heavy right-skewness and stabilize variance.

* **Cyclical Encoding:** Months (1-12) were transformed using Sine and Cosine functions to preserve the temporal proximity between December and January.
* **Volatility Clustering (Lags):** Engineered 1, 2, and 3-month lags for `Volatility_Log` and `Volume` to capture the "long memory" effect in market volatility.
* **Trend Indicators:** Developed 3 and 6-period Simple Moving Averages (SMA) and Exponential Moving Averages (EMA) to smooth noise and highlight momentum.

## 🚀 Modelling & Performance
The models were evaluated using a **Time-Series Split** (80% training / 20% validation) to prevent data leakage. Hyperparameters were optimized via `RandomizedSearchCV`.

### RMSE Results Comparison
| Model | Validation RMSE |
| :--- | :--- |
| **CatBoost Regressor** | **0.2046** |
| **Random Forest** | 0.2519 |
| **LightGBM** | 0.3626 |

### Feature Importance
Across all models, **Exponential Moving Averages (EMA)** and **Lagged Volatility** were the strongest predictors, confirming that recent volatility is the best indicator of future volatility.

## 📁 Project Structure
* `A1_stock_volatility_prediction.ipynb`: The complete end-to-end Python pipeline.
* `A1_stock_volatility_labeled.csv`: The historical training dataset.
* `A1_stock_volatility_submission.csv`: Final model predictions for the 2023-11-01 horizon.

## 💻 Requirements
* Python 3.x
* Pandas, Numpy
* Matplotlib
* Scikit-Learn
  - Random Forest
* CatBoost
* LightGBM
