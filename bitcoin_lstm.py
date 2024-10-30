import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Step 1: Load the Bitcoin data
url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=BTCUSD&apikey=YOUR_API_KEY&datatype=csv"
data = pd.read_csv(url)
data = data.sort_values("timestamp")  # Ensure the data is in chronological order

# Step 2: Preprocessing
data['close'] = data['close'].astype(float)
close_data = data['close'].values.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_data)

# Define training and test set sizes
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Step 3: Create data sequences for LSTM
def create_sequences(data, seq_len):
    x = []
    y = []
    for i in range(seq_len, len(data)):
        x.append(data[i-seq_len:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

seq_len = 60  # The sequence length to look back in time
x_train, y_train = create_sequences(train_data, seq_len)
x_test, y_test = create_sequences(test_data, seq_len)

# Reshape for LSTM layer compatibility
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Step 4: Build the LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Step 5: Train the model
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test), verbose=1)

# Step 6: Make predictions
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
actual_prices = scaler.inverse_transform(y_test.reshape(-1,1))

# Step 7: Visualization
plt.figure(figsize=(14,5))
plt.plot(actual_prices, color='black', label="Actual Bitcoin Price")
plt.plot(predicted_prices, color='green', label="Predicted Bitcoin Price")
plt.title("Bitcoin Price Prediction")
plt.xlabel("Time")
plt.ylabel("Bitcoin Price (USD)")
plt.legend()
plt.show()

# Step 8: Evaluation
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(actual_prices, predicted_prices)
print(f"Mean Squared Error: {mse}")
