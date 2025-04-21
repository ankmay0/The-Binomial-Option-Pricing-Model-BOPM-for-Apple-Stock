import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime, timedelta

# Load model and scaler bundle
model = load_model('model.h5')
bundle = joblib.load('scaler_and_features.joblib')
scaler = bundle['scaler']
features = bundle['features']

# RSI Calculation
def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Data Preparation
def prepare_data(symbol, end_date):
    start_date = (pd.to_datetime(end_date) - timedelta(days=300)).strftime('%Y-%m-%d')
    end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')

    stock_data = yf.download(symbol, start=start_date, end=end_date)
    stock_data['Close'] = stock_data['Close'].fillna(method='ffill')

    stock_data['RSI'] = compute_rsi(stock_data['Close'], 14)
    stock_data['MA5'] = stock_data['Close'].rolling(window=5).mean()
    stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['Volume'] = stock_data['Volume'].fillna(0)

    stock_data = stock_data.dropna()
    scaled_data = scaler.transform(stock_data[features])
    return stock_data, scaled_data

# Prediction
def predict_price(scaled_data):
    sequence_length = 60
    if len(scaled_data) < sequence_length:
        return None
    X = np.array([scaled_data[-sequence_length:]])
    prediction = model.predict(X)
    return prediction[0][0]

# Streamlit UI
st.title("🍎 AAPL Stock Price Predictor")

selected_date = st.date_input(
    "Select a date (historical prediction up to this date):",
    value=datetime(2022, 12, 31),
    min_value=datetime(2020, 1, 1),
    max_value=datetime.today() - timedelta(days=1)
)

symbol = "AAPL"
st.write(f"Fetching AAPL data up to {selected_date}...")

stock_data, scaled_data = prepare_data(symbol, selected_date)

# Show data
st.subheader("📄 Last 5 Rows of Preprocessed Data")
st.write(stock_data.tail())

# Prediction
st.subheader("🔮 Predicted Next Closing Price (Next Trading Day)")
predicted_price = predict_price(scaled_data)
if predicted_price is not None:
    st.success(f"${predicted_price:.2f}")
else:
    st.error("Not enough data to make a prediction. Try a later date.")

# Chart
st.subheader("📈 Closing Price History")
st.line_chart(stock_data['Close'])
