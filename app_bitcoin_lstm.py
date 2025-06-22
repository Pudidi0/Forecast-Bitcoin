
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# ========================== #
# Sidebar Parameter Input
# ========================== #
st.sidebar.header("🔧 Konfigurasi Model")
crypto_symbol = st.sidebar.selectbox("Pilih Crypto", ['BTC-USD', 'ETH-USD', 'DOGE-USD'])
start_date = st.sidebar.date_input("Tanggal Awal", pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input("Tanggal Akhir", pd.to_datetime('2024-12-31'))
prediction_days = st.sidebar.slider("Hari Prediksi", 5, 60, 30)
window_size = st.sidebar.slider("Ukuran Jendela Waktu", 10, 120, 60)
epochs = st.sidebar.slider("Jumlah Epoch", 5, 100, 50)
batch_size = st.sidebar.selectbox("Batch Size", [8, 16, 32])

st.title("📈 Prediksi Harga Bitcoin Menggunakan LSTM")

# ========================== #
# Ambil Data dari Yahoo Finance
# ========================== #
@st.cache_data
def load_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    return data[['Close']].dropna()

df = load_data(crypto_symbol, start_date, end_date)
st.write("Data Historis", df.tail())

# ========================== #
# Preprocessing
# ========================== #
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

X, y = [], []
for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i - window_size:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X).reshape(-1, window_size, 1)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# ========================== #
# Model Training
# ========================== #
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop], verbose=0)

# ========================== #
# Evaluasi
# ========================== #
y_pred_train = scaler.inverse_transform(model.predict(X))
y_actual_train = scaler.inverse_transform(y.reshape(-1, 1))

mae = mean_absolute_error(y_actual_train, y_pred_train)
rmse = np.sqrt(mean_squared_error(y_actual_train, y_pred_train))
r2 = r2_score(y_actual_train, y_pred_train)

st.subheader("📊 Evaluasi Model")
st.write(f"**MAE**  : {mae:.2f}")
st.write(f"**RMSE** : {rmse:.2f}")
st.write(f"**R²**   : {r2:.4f}")

# ========================== #
# Visualisasi Loss
# ========================== #
st.subheader("📉 Grafik Loss selama Training")
fig1, ax1 = plt.subplots()
ax1.plot(history.history['loss'], label='Train Loss')
ax1.plot(history.history['val_loss'], label='Val Loss')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss (MSE)")
ax1.legend()
st.pyplot(fig1)

# ========================== #
# Prediksi Harga Ke Depan
# ========================== #
st.subheader(f"🔮 Prediksi {prediction_days} Hari ke Depan")
last_seq = scaled_data[-window_size:]
future_pred_scaled = []

current_input = last_seq.reshape(1, window_size, 1)
for _ in range(prediction_days):
    pred = model.predict(current_input, verbose=0)[0][0]
    future_pred_scaled.append(pred)
    current_input = np.append(current_input[0][1:], [[pred]], axis=0).reshape(1, window_size, 1)

future_prices = scaler.inverse_transform(np.array(future_pred_scaled).reshape(-1, 1))
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=prediction_days)

pred_df = pd.DataFrame(future_prices, index=future_dates, columns=['Predicted'])

fig2, ax2 = plt.subplots()
ax2.plot(df[-window_size:].index, df['Close'][-window_size:], label='Historis')
ax2.plot(pred_df.index, pred_df['Predicted'], label='Prediksi', linestyle='--', color='red')
ax2.legend()
st.pyplot(fig2)
