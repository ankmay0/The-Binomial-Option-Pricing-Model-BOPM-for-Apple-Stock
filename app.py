import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import joblib

# Load model and scaler
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

# DL Prediction
def predict_price(scaled_data):
    sequence_length = 60
    if len(scaled_data) < sequence_length:
        return None
    X = np.array([scaled_data[-sequence_length:]])
    prediction = model.predict(X)
    return prediction[0][0]

# Binomial Option Pricing Model
def binomial_option_pricing(S, K, T, r, sigma, N, option_type='call'):
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)

    asset_prices = [S * (u ** j) * (d ** (N - j)) for j in range(N + 1)]

    if option_type == 'call':
        option_values = [max(0, price - K) for price in asset_prices]
    else:
        option_values = [max(0, K - price) for price in asset_prices]

    for i in range(N - 1, -1, -1):
        option_values = [
            math.exp(-r * dt) * (p * option_values[j + 1] + (1 - p) * option_values[j])
            for j in range(i + 1)
        ]

    return option_values[0]

# Streamlit UI
st.set_page_config(page_title="Apple Stock Predictor", layout="centered")
st.title(" Apple Stock Predictor + Option Pricing")

selected_date = st.date_input(
    "Select a date (historical prediction up to this date):",
    value=datetime(2022, 12, 31),
    min_value=datetime(2020, 1, 1),
    max_value=datetime.today() - timedelta(days=1)
)

symbol = "AAPL"
st.write(f"Fetching AAPL data up to {selected_date}...")

stock_data, scaled_data = prepare_data(symbol, selected_date)

st.subheader("📄 Last 5 Rows of Preprocessed Data")
st.write(stock_data.tail())

# DL Prediction
st.subheader("🔮 DL Predicted Next Closing Price")
predicted_price_dl = predict_price(scaled_data)

if predicted_price_dl is not None:
    st.success(f"${predicted_price_dl:.2f}")

    # BOPM based on DL price
    st.subheader("📉 Option Pricing based on DL Prediction")

    K = st.number_input("Strike Price (K)", value=float(predicted_price_dl))
    T_days = st.slider("Time to Maturity (in days)", min_value=30, max_value=365, value=90)
    T = T_days / 365
    r = st.number_input("Risk-Free Interest Rate (%)", value=5.0) / 100
    sigma = st.number_input("Volatility (σ)", value=0.25)
    option_type = st.selectbox("Option Type", options=['call', 'put'])

    option_price = binomial_option_pricing(
        S=predicted_price_dl,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        N=100,
        option_type=option_type
    )

    st.success(f"{option_type.title()} Option Price: ${option_price:.2f}")

else:
    st.error("Not enough data to make a prediction. Try a later date.")

# Chart
st.subheader("📈 Closing Price History")
st.line_chart(stock_data['Close'])
