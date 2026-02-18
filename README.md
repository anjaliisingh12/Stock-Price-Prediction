# 📈 Stock Trend Prediction using LSTM

This project predicts stock price trends using **Long Short-Term Memory (LSTM)** deep learning models.  
The app is built using **Python, TensorFlow/Keras, and Streamlit**, allowing users to search for any stock ticker and visualize historical data along with predicted trends.

---

## 🚀 Live Demo

🔗 https://stock-price-prediction-j52k2q8iaxvdxw3uhnive3.streamlit.app

---

## 🧠 Project Overview

Stock prices are sequential time-series data.  
To capture long-term dependencies and patterns, this project uses an **LSTM neural network**, which is highly effective for time-series forecasting.

The model is trained on historical closing prices and predicts future trends.

---

## 📊 Features

- Fetches real-time historical stock data using **Yahoo Finance (yfinance)**
- Visualizes:
  - Closing Price vs Time
  - 100-Day Moving Average
  - 200-Day Moving Average
- Splits data into training (70%) and testing (30%)
- Uses **MinMaxScaler** for normalization
- LSTM-based deep learning model
- Interactive Streamlit web application
- Supports:
  - 🇺🇸 US Stocks (AAPL, MSFT, AMZN, etc.)
  - 🇮🇳 Indian Stocks (TCS.NS, INFY.NS, SBIN.NS, etc.)

---

## 🛠 Tech Stack

- Python
- Streamlit
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- yfinance

---

## 📂 Project Structure

```
Stock-Price-Prediction/
│
├── app.py
├── keras_model.h5
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## ▶️ Run Locally

Clone the repository:

```bash
git clone https://github.com/anjaliisingh12/Stock-Price-Prediction.git
cd Stock-Price-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

---

## 📌 Model Details

- Architecture: LSTM Neural Network
- Input: Previous 100 days of closing prices
- Output: Predicted price trend
- Loss Function: Mean Squared Error
- Data Scaling: MinMaxScaler (0-1 range)

---

## ⚠️ Disclaimer

This project is for educational purposes only.  
It does not provide financial advice.

---

## 👩‍💻 Author

**Anjali Singh**  
B.Tech | AI & ML Enthusiast  

GitHub: https://github.com/anjaliisingh12  

---

⭐ If you like this project, consider giving it a star!
