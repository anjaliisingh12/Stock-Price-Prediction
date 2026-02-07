import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

st.title('Stock Trend Prediction')

import yfinance as yf

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start = '2019-01-01', end = '2024-12-31')
print (df.head())

#Describing Data
st.subheader('Data from 2019 - 2024')
st.write(df.describe())

#visualizations
st.subheader('Closing Price vs Time chart')
fig,ax = plt.subplots(figsize=(12,6))
ax.plot(df['Close'])
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig,ax = plt.subplots(figsize=(12,6))
plt.plot(ma100,label='100MA', color='orange')
ax.plot(df['Close'])
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig,ax = plt.subplots(figsize=(12,6))
ax.plot(df['Close'], label='Closing price')
ax.plot(ma100,label='100MA', color='orange')
ax.plot(ma200,label='200MA', color='green')
ax.legend()
st.pyplot(fig)


# Splitting data into Training and testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)



#load my model
model = load_model("keras_model.h5")


#testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])
    
    
x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler=scaler.scale_

scaler_factor = 1/scaler[0]
y_predicted = y_predicted * scaler_factor
y_test = y_test * scaler_factor


# Final Graph

st.subheader('Predictions Vs Origional')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b' , label = 'Origional Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()

st.pyplot(fig2)
