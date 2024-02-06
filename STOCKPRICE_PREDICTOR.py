import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

file_to_load = 'DATASET.csv'
df = pd.read_csv(file_to_load)

file_to_save = 'stock_data_processed.csv'

if not os.path.isfile(file_to_save):
    df.to_csv(file_to_save)

# If the data is already there, just load it from the CSV
else:
    print('File already exists. Loading data from CSV')
    df = pd.read_csv(file_to_save)

df = df.sort_values('Date')

df.head()

# Calculate the mid prices from the highest and lowest
high_prices = df.loc[:, 'High'].values
low_prices = df.loc[:, 'Low'].values
mid_prices = (high_prices + low_prices) / 2.0

scaler = MinMaxScaler()
mid_prices = scaler.fit_transform(mid_prices.reshape(-1, 1))

split_percentage = 0.8
split_index = int(len(mid_prices) * split_percentage)

# Split the data into training and test sets
train_data = mid_prices[:split_index]
test_data = mid_prices[split_index:]

print("Length of mid_prices:", len(mid_prices))
print("Split index:", split_index)
print("Length of train_data:", len(train_data))
print("Length of test_data:", len(test_data))

# Ensure that test_data is non-empty
if len(test_data) == 0:
    raise ValueError("Test data is empty. Please check the splitting of data.")

# Reshape data for LSTM input
train_data = np.reshape(train_data, (train_data.shape[0], 1, train_data.shape[1]))
test_data = np.reshape(test_data, (test_data.shape[0], 1, test_data.shape[1]))

# Create the LSTM model
model = Sequential([
    LSTM(50, input_shape=(train_data.shape[1], train_data.shape[2])),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(train_data, train_data, epochs=100, batch_size=64)

# Get the predicted prices
predicted_prices = model.predict(test_data)

predicted_prices = np.reshape(predicted_prices, (predicted_prices.shape[0], 1))
predicted_prices = scaler.inverse_transform(predicted_prices)

def plot_stock_price(actual_prices, predicted_prices):
    plt.figure(figsize=(12, 6))
    plt.plot(predicted_prices, label='Predicted Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.title('Stock Price Prediction')
    plt.show()

plot_stock_price(mid_prices[split_index:], predicted_prices)
