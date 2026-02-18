import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Stock Trend Prediction", layout="wide")

st.title("📈 Stock Trend Prediction using LSTM")

# =========================
# User Input
# =========================
user_input = st.text_input("Enter Stock Ticker (Example: AAPL, TCS.NS)", "AAPL")

# =========================
# Download Data
# =========================
df = yf.download(user_input, start="2019-01-01", end="2024-12-31")

if df.empty:
    st.error("Invalid ticker or no data found. Please try another stock.")
    st.stop()

df = df.dropna()

# =========================
# Data Description
# =========================
st.subheader("📊 Data from 2019 - 2024")
st.write(df.describe())

# =========================
# Visualization 1
# =========================
st.subheader("Closing Price vs Time Chart")
fig1 = plt.figure(figsize=(12,6))
plt.plot(df['Close'])
plt.xlabel("Time")
plt.ylabel("Price")
st.pyplot(fig1)

# =========================
# Visualization 2
# =========================
st.subheader("Closing Price with 100 Day Moving Average")
ma100 = df['Close'].rolling(100).mean()

fig2 = plt.figure(figsize=(12,6))
plt.plot(df['Close'], label="Closing Price")
plt.plot(ma100, 'r', label="100 MA")
plt.legend()
st.pyplot(fig2)

# =========================
# Visualization 3
# =========================
st.subheader("Closing Price with 100 & 200 Day Moving Average")
ma200 = df['Close'].rolling(200).mean()

fig3 = plt.figure(figsize=(12,6))
plt.plot(df['Close'], label="Closing Price")
plt.plot(ma100, 'r', label="100 MA")
plt.plot(ma200, 'g', label="200 MA")
plt.legend()
st.pyplot(fig3)

# =========================
# Data Preparation
# =========================
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

# =========================
# Load Model
# =========================
model = load_model("keras_model.h5")

# =========================
# Testing Part
# =========================
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)

y_predicted = model.predict(x_test)

# Reverse Scaling
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# =========================
# Final Plot
# =========================
st.subheader("📉 Prediction vs Original")

fig4 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label="Original Price")
plt.plot(y_predicted, 'r', label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig4)
