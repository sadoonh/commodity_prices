import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import plotly.graph_objs as go
import plotly.io as pio

# Set the renderer to 'notebook_connected' for Jupyter Notebook output
pio.renderers.default = 'notebook_connected'

# Load copper data
copper = yf.Ticker("LAC")
hist = copper.history(period="ytd")

# Use the 'Close' prices for forecasting
data = hist[['Close']].copy()

# Normalize the 'Close' prices
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)

# Define the window size
window_size = 30

# Create dataset
def create_dataset(dataset, window_size):
    X, y = [], []
    for i in range(len(dataset) - window_size - 1):
        a = dataset[i:(i + window_size), 0]
        X.append(a)
        y.append(dataset[i + window_size, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data, window_size)

# Reshape the data to fit the model
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# Build the 1D-CNN model
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(window_size, 1)))
model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(20, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(1))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate the model
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert predictions
train_predict = scaler.inverse_transform(train_predict)
y_train_inv = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test_inv = scaler.inverse_transform([y_test])

# Create a DataFrame for Actual vs Predicted
train_predict_series = pd.Series(train_predict.flatten(), index=data.index[window_size:len(train_predict) + window_size])
test_predict_series = pd.Series(test_predict.flatten(), index=data.index[len(train_predict) + (window_size):len(train_predict) + (window_size) + len(test_predict)])

comparison_df = pd.DataFrame({
    'Actual': data['Close'],
    'Train Predict': train_predict_series,
    'Test Predict': test_predict_series
})

# Forecast future copper prices
future_steps = 30
last_window = X_test[-1]

future_predictions = []
for _ in range(future_steps):
    next_step = model.predict(last_window.reshape(1, window_size, 1))
    future_predictions.append(next_step[0, 0])
    last_window = np.append(last_window[1:], next_step, axis=0)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Create future dates for the prediction, starting immediately after the last actual date
last_date = data.index[-1]
future_dates = pd.date_range(last_date + pd.DateOffset(days=1), periods=future_steps, freq='B')

# Create a series for future predictions
future_predict_series = pd.Series(future_predictions.flatten(), index=future_dates)

# Add future predictions to the comparison DataFrame
future_df = pd.DataFrame({
    'Actual': [np.nan] * len(future_predict_series),
    'Train Predict': [np.nan] * len(future_predict_series),
    'Test Predict': [np.nan] * len(future_predict_series),
    'Future Predict': future_predict_series
})

# Append future_df to comparison_df
comparison_df = pd.concat([comparison_df, future_df])

# Dynamically filter the last 4 months of data
end_date = comparison_df.index.max()
start_date = end_date - pd.DateOffset(months=4)
comparison_df = comparison_df.loc[start_date:end_date]

comparison_df.to_csv('lithium_forecast.csv')

