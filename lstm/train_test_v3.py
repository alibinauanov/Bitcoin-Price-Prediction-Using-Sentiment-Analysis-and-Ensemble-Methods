import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2

# Load and preprocess the data
df = pd.read_csv('dataset_cleaned.csv')

# Drop rows with missing textblob_polarity scores or price (if applicable)
df = df.dropna(subset=['textblob_polarity', 'open'])

# Convert `date` to datetime and sort (optional but useful for time-series data)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.sort_values(by='date')

# Scale the features and target
price_scaler = MinMaxScaler()
textblob_polarity_scaler = MinMaxScaler()

df['open_scaled'] = price_scaler.fit_transform(df[['open']])
df['textblob_polarity_scaled'] = textblob_polarity_scaler.fit_transform(df[['textblob_polarity']])

# Prepare data for LSTM
time_steps = 10  # Look-back period
X, y = [], []

for i in range(len(df) - time_steps):
    X.append(df[['open_scaled', 'textblob_polarity_scaled']].iloc[i:i + time_steps].values)
    y.append(df['open_scaled'].iloc[i + time_steps])

X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Define the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=64, return_sequences=True, input_shape=(time_steps, 2)),  # Increased units
    tf.keras.layers.Dropout(0.3),  # Reduced Dropout
    tf.keras.layers.LSTM(units=64, return_sequences=False),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer=l2(0.01)),
    tf.keras.layers.Dense(units=1)  # Regression output for price prediction
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',         # Monitor the validation loss
    patience=5,                 # Stop if no improvement for 5 epochs
    restore_best_weights=True   # Revert to the best weights during training
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,  # Increased epochs for better learning
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],  # Use early stopping callback
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

# Debug shapes of predictions and ground truth
print("Shape of y_pred (before rescaling):", y_pred.shape)
print("Shape of y_test (before rescaling):", y_test.shape)

# Inverse transform the predictions and actual values
try:
    y_pred_rescaled = price_scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_rescaled = price_scaler.inverse_transform(y_test.reshape(-1, 1))
except Exception as e:
    print("Error during inverse transformation:", e)
    raise

# Debug the first few values
print("First few values of y_pred_rescaled:", y_pred_rescaled[:5].flatten())
print("First few values of y_test_rescaled:", y_test_rescaled[:5].flatten())

# Plot actual vs predicted prices
plt.figure(figsize=(10, 6))

# Ensure arrays are 1D for plotting
plt.plot(y_test_rescaled.flatten(), label='Actual Prices')
plt.plot(y_pred_rescaled.flatten(), label='Predicted Prices')

plt.legend()
plt.title('Actual vs Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Bitcoin Price')
plt.show()
