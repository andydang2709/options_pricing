# OPTIONS PRICING - MACHINE LEARNING ALGORITHMS

## Problem Statement

The Black-Scholes model, while widely used for pricing European call options, often falls short in real-world scenarios, such as volatility skew, interest rate fluctuations, and American-style options. These limitations lead to inaccurate pricing and increased financial risk. Machine learning models, with their ability to capture non-linear patterns and complex dependencies, have the potential to provide more accurate predictions by adapting to these market anomalies. This project aims to develop ML models to improve options price prediction accuracy, offering a more reliable tool for traders and investors.

## Available Data

18.8M+ rows for option and stock prices from S&P 500 companies from 11/08/2021 - 08/28/2024. There are 11 relevant features and 1 target variable - option price.

- The initial date is chosen as the first day after the US reopens its border from COVID-19 restrictions.
- Relevant features include: 'stock_price', 'strike', 'call_put_Call', 'volatility', 'time_to_expiration', 'vol', 'delta', 'gamma', 'theta', 'vega', 'rho'

## Data Mining

Data is collected from public sources such as Wikipedia, Yahoo Finance, and a data repository hosted on Dolthub.
1. Wikipedia: http://en.wikipedia.org/wiki/List_of_S%26P_500_companies
2. Yahoo Finance API: https://pypi.org/project/yfinance/
3. Dolthub: https://www.dolthub.com/repositories/post-no-preference/options/data/master

List of S&P 500 companies' tickers are scraped from Wikipedia, which is then used to extract stock prices from Yahoo Finance's API. The list of tickers are then fed into SQL queries to pull data from Dolthub.
Data was initially loaded in a separate train and test set. They were later joined and re-split into train-test with the sklearn's TimeSeriesSplit function.

## Data Cleaning

- One-hot encode the call_put function and convert into trainable format
- Calculate stock volatility for the time period of the dataset
### Remove uninformative and non-representative option records (Anders, 1996)
- Calculate the option price as the average of the 'bid' and 'ask' columns to avoid bid-ask fluctuations
- Remove options that have less than 15 days to maturity
- Remove options that violate the lower boundary conditions for value of call
- Remove options that are deep-in- or deep-out-of-the-money

## Modeling

### Model Choices:
- Black-Scholes
