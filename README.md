# OPTIONS PRICING - MACHINE LEARNING ALGORITHMS

## Problem Statement

The Black-Scholes model, while widely used for pricing European call options, often falls short in real-world scenarios, such as volatility skew, interest rate fluctuations, and American-style options. These limitations lead to inaccurate pricing and increased financial risk. Machine learning models, with their ability to capture non-linear patterns and complex dependencies, have the potential to provide more accurate predictions by adapting to these market anomalies. This project aims to develop ML models to improve options price prediction accuracy, offering a more reliable tool for traders and investors.

## Available Data

18.8M+ rows for option and stock prices from S&P 500 companies from 11/08/2021 - 08/28/2024. There are 11 relevant features and 1 target variable - option price.

- The initial date is chosen as the first day after the US reopens its border from COVID-19 restrictions.
- Relevant features include: 'stock_price', 'strike', 'call_put_Call', 'volatility', 'time_to_expiration', 'vol', 'delta', 'gamma', 'theta', 'vega', 'rho'

## Assumptions
- **Stable Relationships**: I assume that the relationships between features and option prices remain consistent over time. Although volatility can fluctuate, for this project, I am simplifying by using the average stock volatility across the entire time horizon.
- **No Extreme Events**: The model assumes that extreme market events (e.g., crashes or abnormal volatility spikes) do not significantly affect predictions. The time horizon is chosen assuming that the financial market became more stable post-COVID restrictions.
- **Feature Relevance**: I assume that the selected features sufficiently capture the factors influencing options pricing.
- **Data Quality**: The model assumes the historical data is accurate, complete, and representative of future market conditions.

## Data Mining

Data is collected from public sources such as Wikipedia, Yahoo Finance, and a data repository hosted on Dolthub.
1. Wikipedia: http://en.wikipedia.org/wiki/List_of_S%26P_500_companies
2. Yahoo Finance API: https://pypi.org/project/yfinance/
3. Dolthub: https://www.dolthub.com/repositories/post-no-preference/options/data/master

List of S&P 500 companies' tickers are scraped from Wikipedia, which is then used to extract stock prices from Yahoo Finance's API. The list of tickers are then fed into SQL queries to pull data from Dolthub.

## Exploratory Data Analysis
- Correlation analysis through data visualization
- Summary statistics analysis for sanity check

## Data Wrangling

### Feature Engineering
- One-hot encode the call_put function and convert into trainable format.
- Calculate stock volatility for the time period of the dataset.
- Split into train-test with the sklearn's TimeSeriesSplit function for improved time series forecast.
  
### Remove uninformative and non-representative option records (Anders, 1996)
- Calculate the option price as the average of the 'bid' and 'ask' columns to avoid bid-ask fluctuations.
- Remove options that have less than 15 days to maturity.
- Remove options that violate the lower boundary conditions for the value of the call.
- Remove options that are deep-in- or deep-out-of-the-money.

## Modeling

### Model Choices:
- **Black-Scholes**: Baseline parametric model for comparison
- **Random Forest**: Capture non-linear relationships between features.
- **Extreme Gradient Boosting (XGBoost)**: Handle large datasets and improve predictive performance by through optiomized learning.
- **Light Gradient Boosting (LGBM)**: Analyze intricate dependencies in large, complex options pricing datasets in a fast, memory-efficient way.

### Performance Metric Choice
We decided to choose RMSE as the performance metric of choice
1. **Captures Magnitude of Errors**: RMSE effectively measures the magnitude of prediction errors, providing a clear sense of how far off predictions are from actual prices.
2. **Penalizes Larger Errors**: RMSE gives higher weight to larger errors, which is important in options pricing where significant mispricing can lead to substantial financial losses.
3. **Reflects Prediction Accuracy**: RMSE provides a straightforward metric to evaluate how well the model's predictions align with real-world option prices.

## Results
All 3 Machine Learning models perform better than Black-Scholes model by 81.6% over average

## Reflection
- Black Scholes model's performance could be better if I calculated the running average volatility of stock prices instead of using 1 average volatility. This could be done with more computational power.
- A wider search for best parameters during hyperparameter tuning could potentially improve the model performance.

# Theories and Citations:
1. Anders, U., Korn, O., & Schmitt, C. "Improving the Pricing of Options: A Neural Network Approach."
2. Ivașcu, C.-F. "Option Pricing Using Machine Learning."
