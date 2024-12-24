# Test loss: 0.0037
# Test MAE: 0.0240
# Predicted Polarity: 0.0214

import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy as np

# Load the processed dataset
data_file = 'processed_bitcoin_tweets_with_sentiment.csv'
df = pd.read_csv(data_file)

# Handle missing or invalid data
df = df.dropna(subset=['text'])  # Drop rows where 'text' is NaN
df['text'] = df['text'].astype(str)  # Ensure all values in 'text' are strings

# Parameters
max_words = 10000  # Vocabulary size
max_len = 100  # Maximum length of sequences
embedding_dim = 100  # Embedding vector size

# Prepare the text data
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
padded_sequences = pad_sequences(sequences, maxlen=max_len)

# Prepare the labels (using textblob_polarity as target)
labels = df['textblob_polarity'].values
labels = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential([
    Embedding(max_words, embedding_dim, input_length=max_len),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1, activation='linear')  # Regression for polarity
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# Save the model
model.save('lstm_sentiment_model.h5')

# Predict polarity for all tweets and save results
df['predicted_polarity'] = np.nan  # Initialize a new column for predictions

# Process tweets in batches and predict polarity
batch_size = 1000  # Larger batch size for better performance with large datasets
for i in range(0, len(df), batch_size):
    batch_texts = df['text'].iloc[i:i+batch_size].tolist()  # Get batch of texts
    batch_sequences = pad_sequences(tokenizer.texts_to_sequences(batch_texts), maxlen=max_len)
    batch_predictions = model.predict(batch_sequences, verbose=0).flatten()  # Predict polarity
    df.loc[i:i+batch_size-1, 'predicted_polarity'] = batch_predictions  # Save predictions

# Save the results to a new CSV file
output_file = 'predicted_bitcoin_tweets.csv'
df.to_csv(output_file, index=False)
print(f"Predicted polarities saved to {output_file}")
