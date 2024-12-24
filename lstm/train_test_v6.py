import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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

# Select features (including 'close' for prediction)
features = ['open', 'adjusted_polarity', 'adjusted_subjectivity', 'close']

# Normalize features
scaler = MinMaxScaler()
train_features = scaler.fit_transform(train_data[features])
test_features = scaler.transform(test_data[features])

# --- STEP 3: CREATE SEQUENCES FOR LSTM ---
def create_sequences(data, window_size=10):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, :-1])  # Last 'window_size' rows for features (exclude 'close')
        y.append(data[i, -1])  # 'close' column as target
    return np.array(X), np.array(y)

# Create sequences for training and testing
X_train, y_train = create_sequences(train_features)
X_test, y_test = create_sequences(test_features)

# --- STEP 4: BUILD AND TRAIN LSTM MODEL ---
# Define EarlyStopping and ReduceLROnPlateau callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6)

# Build the model
model = Sequential([
    Bidirectional(LSTM(64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]),
                        kernel_regularizer=regularizers.l2(0.01))),  # L2 regularization
    Dropout(0.3),  # Dropout for regularization
    LSTM(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1)  # Single output for regression
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])

# Train the model with early stopping and learning rate scheduler
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, lr_scheduler],
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
    # Inverse transform for both predictions and actual values
    y_pred_rescaled = scaler.inverse_transform(np.column_stack([np.zeros_like(y_pred), np.zeros_like(y_pred), np.zeros_like(y_pred), y_pred]))[:, -1]
    y_test_rescaled = scaler.inverse_transform(np.column_stack([np.zeros_like(y_test), np.zeros_like(y_test), np.zeros_like(y_test), y_test]))[:, -1]
except Exception as e:
    print("Error during inverse transformation:", e)
    raise

# Debug the first few values of rescaled predictions and actual values
print("First few values of y_pred_rescaled:", y_pred_rescaled[:5])
print("First few values of y_test_rescaled:", y_test_rescaled[:5])

# Get the dates for the test set (make sure the dates are ordered in the same way as the predictions)
test_dates = test_data[test_data['date'] > split_date].iloc[10:]['date'].reset_index(drop=True)

# Plot actual vs predicted prices with timeline on the x-axis
plt.figure(figsize=(10, 6))
plt.plot(test_dates, y_test_rescaled.flatten(), label='Actual Prices')
plt.plot(test_dates, y_pred_rescaled.flatten(), label='Predicted Prices')
plt.legend()
plt.title('Actual vs Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)  # Rotate date labels for better visibility
plt.tight_layout()  # Adjust layout to prevent clipping of labels
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
model.save('new_lstm_32batches_dec5_with_sequences.h5')
