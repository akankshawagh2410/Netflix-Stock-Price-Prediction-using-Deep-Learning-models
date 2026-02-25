# Netflix-Stock-Price-Prediction-using-Deep-Learning-models

This project focuses on forecasting the closing stock price of Netflix (NFLX) using deep learning models designed for time-series prediction. The objective was to predict the next-day closing price for January 2025 using historical stock data from 2019–2024. I implemented and compared multiple neural network architectures to determine which model best captures temporal dependencies in financial data.

## Detail
To understand this project in detail, you can go ahead and read my blog. Here's the link to [My Blog](https://akankshawagh.medium.com/predicting-netflix-stock-prices-using-deep-learning-lstm-gru-and-rnn-compared-419e0cda004b).

## What This Project Demonstrates
- Time-series preprocessing
- Sliding window modeling
- RNN architecture comparison
- Model evaluation & interpretation
- Business understanding of financial forecasting

## Problem Statement
Stock prices are sequential and highly volatile. The goal was to:
- Predict the Close price
- Use historical features: Open, High, Low, Volume
- Forecast on a short-term horizon (next trading day)
This is a supervised time-series forecasting problem.

## Dataset
Source: Historical stock market data (2019–2024)
Features Used:
- Open
- High
- Low
- Volume
- Close (Target variable)

Why These Features?
- OHLC captures intraday price dynamics
- Volume reflects market participation & volatility
- Close price represents final market consensus

## Tech Stack 
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Modeling Approach
Since stock data is sequential, we tested both recurrent and non-recurrent architectures.
**Models Implemented**
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- SimpleRNN
- Dense Neural Network (DNN) – baseline model

## Evaluation Metrics  
We used:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

Why These Metrics?
- RMSE penalizes large financial prediction errors
- R² measures explanatory power
- MAE provides interpretable absolute error


## Results
| Model | RMSE | R² Score |
|----------|-------------|-------------|
| GRU | 0.02075 | 0.9824 |
| LSTM | Slightly higher | High |
| Simple RNN | Moderate | Moderate |
| DNN | Weak | Low |

**Best Model: GRU**
GRU is the best model because it has the *lowest RMSE, highest R² and balanced performance with computational efficiency.*
