# Crypto Stock Price Prediction Bot

This Streamlit-based web application allows users to predict the stock prices of popular cryptocurrencies (BTC, ETH, LTC) for the next 30 days based on historical data from Yahoo Finance. It also provides various features and indicators to assist users in making informed trading decisions.

## Table of Contents

- [Demo](#demo)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Scope of Improvements](#scope-of-improvements)

## Demo

To see the live demo of this crypto trading bot, visit the following link: 
[Crypto Trading Bot Demo](https://intelcryptoguidetradingbot-gv6oli6lrssm4q7qqmczd9.streamlit.app/)

## Features

1. **Crypto Selection**: Choose between BTC, ETH, or LTC as your cryptocurrency of interest.

2. **Historical Data Display**: View the historical price data as a DataFrame.

3. **Monthly Average Prices**: Analyze and visualize the monthly average crypto prices from 2018.

4. **Visual Analysis**: Explore visualizations of stock prices for better insights.

5. **Indicators**:
   - 5.1. Simple Moving Average (SMA)
   - 5.2. Exponential Moving Average (EMA)
   - 5.3. Relative Strength Index (RSI)
   - 5.4. Moving Average Convergence Divergence (MACD), histogram, and signal line.

6. **Predicted Prices**: Get predictions for the next 30 days for the selected cryptocurrency.

7. **Trading Advice**: Receive recommendations on whether to BUY, SELL, or HOLD based on your crypto holdings.

## Getting Started

### Prerequisites

Before running this project, ensure you have the following dependencies installed:

- Python 3.x
- Streamlit
- Pandas
- NumPy
- Matplotlib
- TA-Lib (for technical indicators)

### Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/your-username/crypto-trading-bot.git

3. Install the required Python packages:
   
       pip install -r requirements.txt
4. Run the Streamlit app:

       streamlit run ctb.py


## Scope of Improvements:
**This project has room for improvement in the following areas:**

1. Multivariate Prediction: Extend the prediction model to use multiple input features for more accurate forecasts.

2. Integrated Decision Making: Incorporate technical indicators into the decision-making process for automated BUY-SELL-HOLD recommendations.

3. Sentiment Analysis: Integrate sentiment analysis using historical news data to enhance trading decisions.

   3.1. Collect historical news data through web automation.

   3.2. Perform sentiment analysis on the dataset to derive sentiment-related features.
