import pandas as pd

# Load the dataset
file_path = 'dataset_textblob_sentiment.csv'
data = pd.read_csv(file_path)

# Convert the 'date' column to datetime
data['date'] = pd.to_datetime(data['date'])

# Define the split date
split_date = pd.to_datetime('2023-05-01')

# Split the data into training and testing sets
train_data = data[data['date'] <= split_date]
test_data = data[data['date'] > split_date]

# Save the splits to separate files (optional)
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

# Print split sizes
print(f"Training set size: {train_data.shape[0]}")
print(f"Testing set size: {test_data.shape[0]}")
