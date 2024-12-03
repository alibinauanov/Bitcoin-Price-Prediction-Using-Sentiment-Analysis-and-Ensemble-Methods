import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

# --- STEP 1: LOAD DATA AND HANDLE ERRORS ---
file_path = 'updated_processed_bitcoin_tweets_with_sentiment.csv'

try:
    data = pd.read_csv(
        file_path,
        quotechar='"',
        on_bad_lines='skip',
        engine='python',
    )
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Convert the 'date' column to datetime
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data = data.dropna(subset=['date'])  # Drop rows with invalid dates

# --- STEP 2: SPLIT DATASET ---
split_date = pd.to_datetime('2023-05-01')
train_data = data[data['date'] <= split_date]
test_data = data[data['date'] > split_date]

# Select features
features = ['open', 'adjusted_polarity', 'adjusted_subjectivity']

# Normalize features
scaler = MinMaxScaler()
train_features = scaler.fit_transform(train_data[features + ['close']])  # Include 'close' implicitly
test_features = scaler.transform(test_data[features + ['close']])

# --- STEP 3: UPDATE SEQUENCES (REMOVE TIMESTEPS) ---
# Now the model will take the whole data directly as input without timesteps
X_train = train_features[:, :-1]  # Use all features except 'close' for X
y_train = train_features[:, -1]   # Use 'close' as target
X_test = test_features[:, :-1]    # Use all features except 'close' for X
y_test = test_features[:, -1]     # Use 'close' as target

# --- STEP 4: BUILD AND TRAIN LSTM MODEL ---
# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Build the model
model = Sequential([
    Bidirectional(LSTM(64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1),
                        kernel_regularizer=regularizers.l2(0.01))),  # L2 regularization
    Dropout(0.2),  # Dropout for regularization
    LSTM(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1)  # Single output for regression
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])

# Reshape data for LSTM input
X_train = np.expand_dims(X_train, axis=-1)  # Add an extra dimension for LSTM input (samples, features, 1)
X_test = np.expand_dims(X_test, axis=-1)    # Add an extra dimension for LSTM input (samples, features, 1)

# Train the model with early stopping
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# --- STEP 5: EVALUATE AND TEST THE MODEL ---
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss (MSE): {test_loss}")
print(f"Test Mean Absolute Error (MAE): {test_mae}")

# Make predictions
y_pred = model.predict(X_test).flatten()

# --- STEP 6: EVALUATION AND VISUALIZATION ---
# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Inverse transform the predictions and actual values
try:
    # Create dummy input array with the same shape as original data (with 3 features)
    # We now include all 4 features as in the scaler (open, adjusted_polarity, adjusted_subjectivity, close)
    y_pred_input = np.column_stack([np.zeros_like(y_pred), np.zeros_like(y_pred), np.zeros_like(y_pred), y_pred])
    y_test_input = np.column_stack([np.zeros_like(y_test), np.zeros_like(y_test), np.zeros_like(y_test), y_test])

    # Inverse transform for both predictions and actual values
    y_pred_rescaled = scaler.inverse_transform(y_pred_input)[:, -1]  # Extract only 'close' column
    y_test_rescaled = scaler.inverse_transform(y_test_input)[:, -1]  # Extract only 'close' column
except Exception as e:
    print("Error during inverse transformation:", e)
    raise

# Debug the first few values of rescaled predictions and actual values
print("First few values of y_pred_rescaled:", y_pred_rescaled[:5])
print("First few values of y_test_rescaled:", y_test_rescaled[:5])

# Plot actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled.flatten(), label='Actual Prices')
plt.plot(y_pred_rescaled.flatten(), label='Predicted Prices')
plt.legend()
plt.title('Actual vs Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()

# Calculate evaluation metrics
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# Plot residuals
residuals = y_test_rescaled.flatten() - y_pred_rescaled.flatten()
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, alpha=0.7)
plt.title('Residual Distribution')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.show()

# Save the model
model.save('new_lstm_32batches.h5')
