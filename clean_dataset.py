import pandas as pd

file_path = 'dataset_textblob_sentiment.csv'
cleaned_file_path = 'dataset_cleaned.csv'

# Attempt to load and preprocess the dataset
try:
    # Load the CSV file, ignoring problematic lines
    data = pd.read_csv(
        file_path,
        quotechar='"',        # Handles quoted text
        engine='python',      # Use Python engine for flexibility
        on_bad_lines='skip'   # Skip problematic lines
    )
    
    # Drop rows with invalid data or errors
    data.dropna(inplace=True)  # Remove rows with NaN values
    
    # Save the cleaned dataset to a new file
    data.to_csv(cleaned_file_path, index=False)
    print(f"Cleaned dataset saved to: {cleaned_file_path}")
except Exception as e:
    print(f"Error cleaning file: {e}")
