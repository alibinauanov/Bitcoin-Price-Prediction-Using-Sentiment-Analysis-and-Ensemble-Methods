import pandas as pd

# Load the CSV data into a DataFrame
data = 'dataset_cleaned.csv'
df = pd.read_csv(data)

# Display the DataFrame before deletion
print("Before deletion:")
print(df.head())  # Print the first few rows to verify

# Delete the 'textblob_polarity' and 'textblob_subjectivity' columns
df.drop(['textblob_polarity', 'textblob_subjectivity'], axis=1, inplace=True)

# Display the DataFrame after deletion
print("\nAfter deletion:")
print(df.head())  # Print the first few rows to verify

# Save the updated DataFrame to a new CSV file
df.to_csv('updated_dataset_cleaned.csv', index=False)

print("\nDone! The updated dataset is saved as 'updated_dataset_cleaned.csv'.")
