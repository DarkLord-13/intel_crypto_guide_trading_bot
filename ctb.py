# -*- coding: utf-8 -*-
"""CryptoGuide-TradingBot (Task-7).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wQv1i5P6TQ3WkSt8e47hjRN2w86G4Ew1
"""
# First we will import the necessary Library

import os
import pandas as pd
import numpy as np
import math
import datetime as dt
import matplotlib.pyplot as plt

# For Evalution we will use these library

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

# For model building we will use these library

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM

# For PLotting we will use these library

import matplotlib.pyplot as plt
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import yfinance as yf
#from IPython.display import display
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Crypto Trading Bot xxx')


st.subheader('Select your crypto-currency:')
symbol = st.selectbox('',['BTC', 'ETH', 'LTC'])


symbol+='-USD'
st.subheader(symbol)

start_date = "2018-01-01"
end_date = dt.datetime.now()

df = yf.download(symbol, start=start_date, end=end_date)
df.to_csv(symbol)
df = pd.read_csv(symbol)

st.dataframe(df)

df = df.dropna(how='any')


df = df.drop(columns= ['Volume', 'Adj Close'], axis=1)

st.write('Monthvise average crypto price from 2018')

df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
monthvise= df.groupby(df['Date'].dt.strftime('%B'))[['Open','Close']].mean()
new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
             'September', 'October', 'November', 'December']
monthvise = monthvise.reindex(new_order, axis=0)
st.table(monthvise) 

fig = go.Figure()

fig.add_trace(go.Bar(
    x=monthvise.index,
    y=monthvise['Open'],
    name='Stock Open Price',
    marker_color='crimson'
))
fig.add_trace(go.Bar(
    x=monthvise.index,
    y=monthvise['Close'],
    name='Stock Close Price',
    marker_color='lightsalmon'
))

fig.update_layout(barmode='group', xaxis_tickangle=-45,
                  title='Monthwise comparision between Stock open and close price')
st.plotly_chart(fig) # display

names = cycle(['Stock Open Price','Stock Close Price','Stock High Price','Stock Low Price'])

fig = px.line(df, x=df.Date, y=[df['Open'], df['Close'],
                                          df['High'], df['Low']],
             labels={'Date': 'Date','value':'Stock value'})
fig.update_layout(title_text='Stock analysis chart', font_size=15, font_color='black',legend_title_text='Stock Parameters')
fig.for_each_trace(lambda t:  t.update(name = next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)

st.plotly_chart(fig)


df = df[['Date', 'Close']]

fig = px.line(df, x=df.Date, y=df.Close,labels={'date':'Date','close':'Close Stock'})
fig.update_traces(marker_line_width=2, opacity=1, marker_line_color='red')
fig.update_layout(title_text=f'Whole period of timeframe of {symbol} close price 2018-2023', plot_bgcolor='white',
                  font_size=15, font_color='black')
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
st.plotly_chart(fig)


df1 = pd.Series(df['Close'])

# Calculate Simple Moving Average (SMA)
sma_window = 100  # Number of data points to use for the SMA window
sma = df1.rolling(window=sma_window).mean()

# Calculate Exponential Moving Average (EMA)
ema_span = 100  # Span for EMA calculation (similar to window for SMA)
ema = df1.ewm(span=ema_span, adjust=False).mean()

#super trend,vwap, rsi, 4 parameters, ema(9days ma, 20, 50, 200)




fig = px.line(y=sma, x=df.Date, title='Simple Moving Average', labels={'x': 'Date', 'y': 'Closing Price'})
st.plotly_chart(fig)

fig = px.line(y=ema, x=df.Date, title='Exponential Moving Average', labels={'x': 'Date', 'y': 'Closing Price'})
st.plotly_chart(fig)


def calculate_rsi(prices, window_length=14):
    delta = np.diff(prices)
    gain = np.where(delta >= 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.convolve(gain, np.ones(window_length)/window_length, mode='valid')
    avg_loss = np.convolve(loss, np.ones(window_length)/window_length, mode='valid')
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate((np.zeros(len(prices) - len(rsi)), rsi))

rsi = calculate_rsi(np.array(df1), window_length=60)

df1 = pd.Series(rsi)

# Calculate Exponential Moving Average (EMA)
ema_span = 100  # Span for EMA calculation (similar to window for SMA)
rsi_ema = df1.ewm(span=ema_span, adjust=False).mean()
fig = px.line(x=df.Date, y=rsi_ema,
              labels={'value': 'Values', 'index': 'Date'},
              title='Relative Strength Index(RSI)-EMA applied')
fig.update_layout(legend_title_text='Columns')
st.plotly_chart(fig)

print('RSI > 70 -> OVERBOUGHT')
print('RSI < 70 -> OVERSOLD')


closing_series = pd.Series(df['Close'])

# Calculate MACD
def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    exp12 = prices.ewm(span=fast_period, adjust=False).mean()
    exp26 = prices.ewm(span=slow_period, adjust=False).mean()
    macd_line = exp12 - exp26
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

macd_line, signal_line, histogram = calculate_macd(closing_series)

# Plotting
plt.figure(figsize=(15, 6))
plt.plot(df['Date'], macd_line, label='MACD Line', color='blue')
plt.plot(df['Date'], signal_line, label='Signal Line', color='red')
plt.bar(df['Date'], histogram, label='Histogram', color='black', alpha=0.5)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('MACD, Signal Line, and Histogram')
plt.legend()
plt.grid(True)
st.pyplot()


df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

df = df.loc[(df['Date'] >= '2021-01-01')
                     & (df['Date'] < dt.datetime.now())]
# dropping the unecessary columns
df = df.drop(columns=['Date'])


scaler=MinMaxScaler(feature_range=(0,1))
df=scaler.fit_transform(np.array(df).reshape(-1,1))


training_size = int(len(df)*0.60)
test_size = len(df)-training_size
train_data,test_data=df[0:training_size,:],df[training_size:len(df),:1]


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 30
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


regressor=Sequential()

regressor.add(LSTM(128,return_sequences=True,input_shape=(X_train.shape[1],1),activation="relu"))
regressor.add(LSTM(64, return_sequences=False))
regressor.add(Dense(25))
regressor.add(Dense(1))

regressor.compile(loss="mean_squared_error",optimizer="adamax")
history = regressor.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=60,batch_size=32,verbose=0)


x_input=test_data[len(test_data)-time_step-1:].reshape(1,-1)

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=time_step
i=0
pred_days = 30 # to predict next 30 days
while(i<pred_days):

    if(len(temp_input)>time_step):

        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))

        yhat = regressor.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)

        lst_output.extend(yhat.tolist())
        i=i+1

    else:

        x_input = x_input.reshape((1, n_steps,1))
        yhat = regressor.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())

        lst_output.extend(yhat.tolist())
        i=i+1

last_days=np.arange(1,time_step+1)
day_pred=np.arange(time_step+1,time_step+pred_days+1)

temp_mat = np.empty((len(last_days)+pred_days+1,1))
temp_mat[:] = np.nan
temp_mat = temp_mat.reshape(1,-1).tolist()[0]

last_original_days_value = temp_mat
next_predicted_days_value = temp_mat

last_original_days_value[0:time_step+1] = scaler.inverse_transform(df[len(df)-time_step:]).reshape(1,-1).tolist()[0]
next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

mid_mean = (sum(last_original_days_value))/len(last_original_days_value)
last_original_days_value[time_step]=mid_mean

new_pred_plot = pd.DataFrame({
    'last_original_days_value':last_original_days_value,
    'next_predicted_days_value':next_predicted_days_value
})

for i in range(time_step):
    new_pred_plot['next_predicted_days_value'][i] = None

for i in range(31, time_step * 2 + 1):  # Fixed the syntax issue here
    new_pred_plot['last_original_days_value'][i] = None


names = cycle(['Last 30 days close price','Predicted next 30 days close price'])

fig = px.line(new_pred_plot,x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                      new_pred_plot['next_predicted_days_value']],
              labels={'value': 'Stock price','index': 'Timestamp'})
fig.update_layout(title_text='Compare last 30 days vs next 30 days',
                  plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')

fig.for_each_trace(lambda t:  t.update(name = next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
st.plotly_chart(fig)

import datetime

def get_next_30_days():
    today = datetime.date.today()
    next_30_days = [today + datetime.timedelta(days=i) for i in range(1, 31)]
    return next_30_days

dates = get_next_30_days()

final_output = pd.DataFrame({
    'DATE':dates,
    'PREDICTED STOCK PRICE(USD)':next_predicted_days_value[31:]
})

st.dataframe(final_output)

max_index = final_output['PREDICTED STOCK PRICE(USD)'].idxmax()
min_index = final_output['PREDICTED STOCK PRICE(USD)'].idxmin()

# Create a DataFrame with max and min information
result_df = pd.DataFrame({
    'Type': ['Maximum', 'Minimum'],
    'Date': [final_output.loc[max_index, 'DATE'], final_output.loc[min_index, 'DATE']],
    'Value (USD)': [final_output.loc[max_index, 'PREDICTED STOCK PRICE(USD)'],
                     final_output.loc[min_index, 'PREDICTED STOCK PRICE(USD)']]
})

st.dataframe(result_df)

st.subheader("To predict whether the user should buy or sell his/her coins")

user_stock_price = st.number_input('Total Bitcoin holdings value(in USD):')
user_stock_coins = st.number_input('Total no. of Bitcoins you own:')

if user_stock_coins>0:
    value_of_one_coin = user_stock_price/user_stock_coins

    max_predicted_price = result_df['Value (USD)'][0]
    min_predicted_price = result_df['Value (USD)'][1]

    if(max_predicted_price>value_of_one_coin):
        st.write('SUGGESTION:  BUY NOW')
        st.write('SELL ON: ', result_df['Date'][0])
    else:
        st.write('HOLD as prices are not going up in next 30 days')
