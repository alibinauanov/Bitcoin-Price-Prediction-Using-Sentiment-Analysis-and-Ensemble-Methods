import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2

# Load and preprocess the data
# df = pd.read_csv('dataset_cleaned.csv')
df = pd.read_csv('dataset_textblob_sentiment.csv')

# Drop rows with missing values in relevant columns
df = df.dropna(subset=['open', '?', 'textblob_polarity', 'textblob_subjectivity'])

# Convert `date` to datetime and sort (optional but useful for time-series data)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.sort_values(by='date')

# Scale the features and target
scalers = {
    'open': MinMaxScaler(),
    'high': MinMaxScaler(),
    'low': MinMaxScaler(),
    'volume': MinMaxScaler(),
    'textblob_polarity': MinMaxScaler(),
    'textblob_subjectivity': MinMaxScaler(),
    'close': MinMaxScaler()  # Target variable
}

# Apply MinMax scaling to each column
for feature, scaler in scalers.items():
    df[f'{feature}_scaled'] = scaler.fit_transform(df[[feature]])

# Prepare data for LSTM
time_steps = 10  # Look-back period
features = [
    'open_scaled',
    'high_scaled',
    'low_scaled',
    'volume_scaled',
    'textblob_polarity_scaled',
    'textblob_subjectivity_scaled'
]
X, y = [], []

for i in range(len(df) - time_steps):
    X.append(df[features].iloc[i:i + time_steps].values)
    y.append(df['close_scaled'].iloc[i + time_steps])

X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Define the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=64, return_sequences=True, input_shape=(time_steps, len(features))),  # Use the number of features
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(units=64, return_sequences=False),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer=l2(0.01)),
    tf.keras.layers.Dense(units=1)  # Regression output for predicting 'close' prices
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='mse', metrics=['mae'])

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Predict and inverse transform
y_pred = model.predict(X_test)

# Inverse transform the predictions and actual values for the 'close' column
close_scaler = scalers['close']  # Target scaler
y_pred_rescaled = close_scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test_rescaled = close_scaler.inverse_transform(y_test.reshape(-1, 1))

# Debug the first few values
print("First few values of y_pred_rescaled:", y_pred_rescaled[:5].flatten())
print("First few values of y_test_rescaled:", y_test_rescaled[:5].flatten())

# Plot actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled.flatten(), label='Actual Prices', linestyle='-', linewidth=2)
plt.plot(y_pred_rescaled.flatten(), label='Predicted Prices', linestyle='--', linewidth=2)
plt.legend()
plt.title('Actual vs Predicted Bitcoin Prices')
plt.xlabel('Time Steps (Test Data)')
plt.ylabel('Bitcoin Price')
plt.show()
