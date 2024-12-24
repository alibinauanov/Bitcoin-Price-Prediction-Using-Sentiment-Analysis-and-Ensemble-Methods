import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from textblob import TextBlob

# --- STEP 1: LOAD DATA AND HANDLE ERRORS ---

file_path = 'dataset_cleaned.csv'

try:
    # Load the CSV file with on_bad_lines to skip problematic lines
    data = pd.read_csv(
        file_path,
        quotechar='"',       # Handles quoted text
        on_bad_lines='skip', # Skip problematic lines
        engine='python',     # Use Python engine for flexibility
    )
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Convert the 'date' column to datetime
data['date'] = pd.to_datetime(data['date'], errors='coerce')  # Handle invalid dates
data = data.dropna(subset=['date'])  # Drop rows with invalid dates
# check the Tensorflow LSTM

# data['textblob_polarity'] = data['tweets'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
# data['textblob_subjectivity'] = data['tweets'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)

# --- STEP 2: SPLIT DATASET ---

# Define split date
split_date = pd.to_datetime('2023-05-01')
train_data = data[data['date'] <= split_date]
test_data = data[data['date'] > split_date]

# Select features and target
features = ['open', 'high', 'low', 'volume', 'textblob_polarity', 'textblob_subjectivity', 'close']
target = 'close'

# Normalize features
scaler = MinMaxScaler()
train_features = scaler.fit_transform(train_data[features])
test_features = scaler.transform(test_data[features])

train_target = train_data[target].values
test_target = test_data[target].values

# --- STEP 3: CREATE SEQUENCES FOR LSTM ---

timesteps = 10

def create_sequences(features, target, timesteps):
    X, y = [], []
    for i in range(len(features) - timesteps):
        X.append(features[i:i+timesteps])
        y.append(target[i+timesteps])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_features, train_target, timesteps)
X_test, y_test = create_sequences(test_features, test_target, timesteps)

# --- STEP 4: BUILD AND TRAIN LSTM MODEL ---

# Define LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(timesteps, len(features))),
    Dense(1)  # Single output for regression
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,  # Adjust epochs as needed
    batch_size=32,
    validation_split=0.2,  # Use 20% of training data for validation
    verbose=1
)

# --- STEP 5: EVALUATE AND TEST THE MODEL ---

# Evaluate on test data
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss (MSE): {test_loss}")
print(f"Test Mean Absolute Error (MAE): {test_mae}")

# Make predictions
predictions = model.predict(X_test).flatten()

# --- STEP 6: EVALUATION AND VISUALIZATION ---

# Calculate metrics
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual', alpha=0.7)
plt.plot(predictions, label='Predicted', alpha=0.7)
plt.title('Actual vs Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Target Value')
plt.legend()
plt.show()

# Plot residuals
residuals = y_test - predictions
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, alpha=0.7)
plt.title('Residual Distribution')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.show()

# --- OPTIONAL: SAVE THE MODEL ---

model.save('lstm_model_50epochs.h5')