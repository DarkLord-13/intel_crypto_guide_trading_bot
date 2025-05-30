# Import necessary libraries
import os
import pandas as pd
import numpy as np
import math
import datetime as dt
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# For Evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.preprocessing import MinMaxScaler

# For model building
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam

# For Plotting
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import yfinance as yf
import streamlit as st

# Configure Streamlit
st.set_page_config(page_title="Crypto Trading Bot", layout="wide")
st.title('🚀 Advanced Crypto Trading Bot Dashboard')

# Sidebar for user inputs
st.sidebar.header("Configuration")

# Crypto selection
st.sidebar.subheader('Select Cryptocurrency:')
crypto_options = {
    'Bitcoin': 'BTC-USD',
    'Ethereum': 'ETH-USD',
    'Litecoin': 'LTC-USD',
    'Cardano': 'ADA-USD',
    'Solana': 'SOL-USD',
    'Dogecoin': 'DOGE-USD'
}
selected_crypto = st.sidebar.selectbox('Choose cryptocurrency:', list(crypto_options.keys()))
symbol = crypto_options[selected_crypto]

# Portfolio inputs
st.sidebar.subheader('Portfolio Information')
holding_value = st.sidebar.number_input('Total Holdings Value (USD):', min_value=0.0, value=0.0, step=100.0)
crypto_name = selected_crypto.split('-')[0] if '-' in selected_crypto else selected_crypto
num_coins = st.sidebar.number_input(f'Number of {crypto_name} coins:', min_value=0.0, value=0.0, step=0.1)

# Date range selection
st.sidebar.subheader('Date Range')
start_date = st.sidebar.date_input('Start Date', dt.date(2020, 1, 1))
end_date = st.sidebar.date_input('End Date', dt.date.today())

# Model parameters
st.sidebar.subheader('Model Parameters')
lstm_units_1 = st.sidebar.slider('First LSTM Units', 32, 256, 128)
lstm_units_2 = st.sidebar.slider('Second LSTM Units', 16, 128, 64)
epochs = st.sidebar.slider('Training Epochs', 10, 100, 50)
batch_size = st.sidebar.slider('Batch Size', 16, 64, 32)
prediction_days = st.sidebar.slider('Prediction Days', 7, 60, 30)


# Data loading and preprocessing
@st.cache_data
def load_data(symbol, start_date, end_date):
    """Load cryptocurrency data from Yahoo Finance"""
    try:
        df = yf.download(symbol, start=start_date, end=end_date)
        if df.empty:
            st.error(f"No data found for {symbol}")
            return None

        # Reset index to make Date a column
        df.reset_index(inplace=True)

        # Ensure we have the expected columns
        expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if 'Adj Close' in df.columns:
            df = df.drop('Adj Close', axis=1)

        # Rename columns if they have different names (MultiIndex issue)
        if hasattr(df.columns, 'levels'):  # MultiIndex columns
            df.columns = df.columns.droplevel(1)  # Remove second level

        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


# Technical indicators
def calculate_sma(data, window):
    """Calculate Simple Moving Average"""
    return data.rolling(window=window).mean()


def calculate_ema(data, span):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=span, adjust=False).mean()


def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    exp1 = prices.ewm(span=fast).mean()
    exp2 = prices.ewm(span=slow).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def create_lstm_dataset(data, time_step=30):
    """Create dataset for LSTM model"""
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)


def build_lstm_model(X_train, lstm1_units=128, lstm2_units=64):
    """Build and compile LSTM model"""
    model = Sequential([
        LSTM(lstm1_units, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='relu'),
        Dropout(0.2),
        LSTM(lstm2_units, return_sequences=False, activation='relu'),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model


# Main application
if st.sidebar.button('🔄 Load Data & Analyze'):
    # Load data
    with st.spinner('Loading cryptocurrency data...'):
        df = load_data(symbol, start_date, end_date)

    if df is not None:
        st.success(f'✅ Data loaded successfully for {selected_crypto}')

        # Display basic info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            current_price = float(df['Close'].iloc[-1])
            st.metric("Current Price", f"${current_price:.2f}")
        with col2:
            current_price_val = float(df['Close'].iloc[-1])
            previous_price_val = float(df['Close'].iloc[-2])
            price_change = current_price_val - previous_price_val
            price_change_pct = (price_change / previous_price_val) * 100
            st.metric("24h Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
        with col3:
            volume = float(df['Volume'].iloc[-1])
            st.metric("Volume", f"{volume:,.0f}")
        with col4:
            market_high = float(df['High'].max())
            st.metric("Market High", f"${market_high:.2f}")

        # Data preview
        st.subheader('📊 Data Overview')
        st.dataframe(df.tail(10))

        # Monthly analysis
        st.subheader('📈 Monthly Price Analysis')
        df['Date'] = pd.to_datetime(df['Date'])

        # Check available columns and handle different column names
        available_cols = df.columns.tolist()
        open_col = 'Open' if 'Open' in available_cols else df.columns[1]  # First price column
        close_col = 'Close' if 'Close' in available_cols else df.columns[4]  # Typically close is 5th column

        # Create monthly grouping with available columns
        monthly_data = df.groupby(df['Date'].dt.strftime('%B'))[[open_col, close_col]].mean()
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        monthwise = monthly_data.reindex([m for m in month_order if m in monthly_data.index])

        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Bar(x=monthwise.index, y=monthwise[open_col],
                                     name='Open Price', marker_color='lightblue'))
        fig_monthly.add_trace(go.Bar(x=monthwise.index, y=monthwise[close_col],
                                     name='Close Price', marker_color='darkblue'))
        fig_monthly.update_layout(barmode='group', title='Monthly Average Prices',
                                  xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig_monthly, use_container_width=True)

        # Price chart
        st.subheader('📈 Price Analysis')
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=df['Date'], y=df['Open'], name='Open', line=dict(color='blue')))
        fig_price.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close', line=dict(color='red')))
        fig_price.add_trace(go.Scatter(x=df['Date'], y=df['High'], name='High', line=dict(color='green')))
        fig_price.add_trace(go.Scatter(x=df['Date'], y=df['Low'], name='Low', line=dict(color='orange')))
        fig_price.update_layout(title=f'{selected_crypto} Price Chart', height=500)
        st.plotly_chart(fig_price, use_container_width=True)

        # Technical indicators
        st.subheader('🔧 Technical Indicators')

        # Calculate indicators
        df['SMA_50'] = calculate_sma(df['Close'], 50)
        df['SMA_200'] = calculate_sma(df['Close'], 200)
        df['EMA_20'] = calculate_ema(df['Close'], 20)
        df['RSI'] = calculate_rsi(df['Close'])
        macd_line, signal_line, histogram = calculate_macd(df['Close'])

        # Moving averages chart
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price', line=dict(color='black')))
        fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], name='SMA 50', line=dict(color='blue')))
        fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['SMA_200'], name='SMA 200', line=dict(color='red')))
        fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['EMA_20'], name='EMA 20', line=dict(color='green')))
        fig_ma.update_layout(title='Moving Averages', height=400)
        st.plotly_chart(fig_ma, use_container_width=True)

        # RSI chart
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='purple')))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        fig_rsi.update_layout(title='Relative Strength Index (RSI)', height=300, yaxis_range=[0, 100])
        st.plotly_chart(fig_rsi, use_container_width=True)

        # MACD chart
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df['Date'], y=macd_line, name='MACD Line', line=dict(color='blue')))
        fig_macd.add_trace(go.Scatter(x=df['Date'], y=signal_line, name='Signal Line', line=dict(color='red')))
        fig_macd.add_trace(go.Bar(x=df['Date'], y=histogram, name='Histogram', marker_color='gray', opacity=0.6))
        fig_macd.update_layout(title='MACD Indicator', height=400)
        st.plotly_chart(fig_macd, use_container_width=True)

        # ML Model Training and Prediction
        st.subheader('🤖 LSTM Price Prediction')

        with st.spinner('Training LSTM model...'):
            # Prepare data for LSTM
            close_prices = df[['Close']].values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(close_prices)

            # Split data
            training_size = int(len(scaled_data) * 0.8)
            train_data = scaled_data[:training_size]
            test_data = scaled_data[training_size:]

            # Create datasets
            time_step = 30
            X_train, y_train = create_lstm_dataset(train_data, time_step)
            X_test, y_test = create_lstm_dataset(test_data, time_step)

            # Reshape for LSTM
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            # Build and train model
            model = build_lstm_model(X_train, lstm_units_1, lstm_units_2)

            # Training progress
            progress_bar = st.progress(0)
            status_text = st.empty()


            class StreamlitCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    status_text.text(f'Training... Epoch {epoch + 1}/{epochs} - Loss: {logs["loss"]:.4f}')


            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=[StreamlitCallback()]
            )

            progress_bar.empty()
            status_text.empty()

        st.success('✅ Model training completed!')

        # Make predictions
        with st.spinner('Generating predictions...'):
            # Predict future prices
            last_sequence = scaled_data[-time_step:]
            predictions = []

            for _ in range(prediction_days):
                next_pred = model.predict(last_sequence.reshape(1, time_step, 1), verbose=0)
                predictions.append(next_pred[0, 0])
                last_sequence = np.roll(last_sequence, -1)
                last_sequence[-1] = next_pred

            # Inverse transform predictions
            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

            # Create future dates
            last_date = df['Date'].iloc[-1]
            future_dates = [last_date + dt.timedelta(days=i) for i in range(1, prediction_days + 1)]

            # Create prediction DataFrame
            prediction_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Price': predictions.flatten()
            })

        # Display predictions
        st.subheader('🔮 Price Predictions')

        # Prediction chart
        fig_pred = go.Figure()

        # Historical data (last 60 days)
        recent_data = df.tail(60)
        fig_pred.add_trace(go.Scatter(
            x=recent_data['Date'],
            y=recent_data['Close'],
            name='Historical Price',
            line=dict(color='blue')
        ))

        # Predictions
        fig_pred.add_trace(go.Scatter(
            x=prediction_df['Date'],
            y=prediction_df['Predicted_Price'],
            name=f'Predicted Price ({prediction_days} days)',
            line=dict(color='red', dash='dash')
        ))

        fig_pred.update_layout(
            title=f'{selected_crypto} Price Prediction',
            height=500,
            xaxis_title='Date',
            yaxis_title='Price (USD)'
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # Prediction table
        st.dataframe(prediction_df)

        # Investment recommendation
        st.subheader('💡 Investment Recommendation')

        current_price = float(df['Close'].iloc[-1])
        max_predicted = float(prediction_df['Predicted_Price'].max())
        min_predicted = float(prediction_df['Predicted_Price'].min())
        avg_predicted = float(prediction_df['Predicted_Price'].mean())

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        with col2:
            st.metric("Predicted Max", f"${max_predicted:.2f}",
                      f"{((max_predicted / current_price - 1) * 100):.1f}%")
        with col3:
            st.metric("Predicted Avg", f"${avg_predicted:.2f}",
                      f"{((avg_predicted / current_price - 1) * 100):.1f}%")

        # Investment advice
        if avg_predicted > current_price * 1.05:  # 5% threshold
            st.success("🟢 **RECOMMENDATION: BUY** - Model predicts price increase")
            max_date = prediction_df.loc[prediction_df['Predicted_Price'].idxmax(), 'Date']
            st.info(f"📅 Consider selling around: {max_date.strftime('%Y-%m-%d')}")
        elif avg_predicted < current_price * 0.95:  # 5% threshold
            st.error("🔴 **RECOMMENDATION: SELL** - Model predicts price decrease")
        else:
            st.warning("🟡 **RECOMMENDATION: HOLD** - Price expected to remain stable")

        # Portfolio calculator
        st.subheader('💰 Portfolio Analysis')

        if holding_value > 0 and num_coins > 0:
            avg_price_paid = float(holding_value / num_coins)
            potential_profit = (max_predicted - avg_price_paid) * num_coins
            potential_loss = (min_predicted - avg_price_paid) * num_coins

            # Display portfolio metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Price Paid", f"${avg_price_paid:.2f}")
            with col2:
                st.metric("Potential Max Profit", f"${potential_profit:.2f}",
                          f"{((potential_profit / holding_value) * 100):.1f}%")
            with col3:
                st.metric("Potential Max Loss", f"${potential_loss:.2f}",
                          f"{((potential_loss / holding_value) * 100):.1f}%")

            # Portfolio recommendation
            if avg_price_paid > 0:
                if max_predicted > avg_price_paid * 1.05:
                    st.success(
                        f"🟢 **Good Entry Point!** Your average price of ${avg_price_paid:.2f} is below predicted maximum of ${max_predicted:.2f}")
                elif avg_price_paid > max_predicted:
                    st.error(
                        f"🔴 **Consider Selling** Your average price of ${avg_price_paid:.2f} is above predicted maximum of ${max_predicted:.2f}")
                else:
                    st.warning(f"🟡 **Monitor Closely** Your position is near predicted levels")
        else:
            st.info("👈 Please enter your portfolio information in the sidebar to see personalized analysis")

# Initialize the app
st.sidebar.markdown("---")
st.sidebar.markdown("### Instructions")
st.sidebar.markdown("1. Select your preferred cryptocurrency")
st.sidebar.markdown("2. Choose date range for analysis")
st.sidebar.markdown("3. Adjust model parameters if needed")
st.sidebar.markdown("4. Click 'Load Data & Analyze' to start")

if 'data_loaded' not in st.session_state:
    st.info("👈 Please configure settings in the sidebar and click 'Load Data & Analyze' to begin.")
