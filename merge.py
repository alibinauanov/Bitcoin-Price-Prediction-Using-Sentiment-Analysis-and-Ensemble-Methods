import dask.dataframe as dd

def merge_bitcoin_tweet_data(file1_path, file2_path, output_path):
    """
    Merges two large files efficiently, optimizing for memory usage.

    Parameters:
        file1_path (str): Path to the first file.
        file2_path (str): Path to the second file.
        output_path (str): Path to save the merged file.

    Returns:
        None: Saves the merged file to the specified output path.
    """
    # Step 1: Load data with specified dtypes
    print("Loading data using Dask with specified dtypes...")
    dtypes = {
        'quote_asset_volume': 'float64',
        'taker_buy_quote_asset_volume': 'float64'
    }
    ddf1 = dd.read_csv(file1_path, dtype=dtypes)  # File with VADER metrics
    ddf2 = dd.read_csv(file2_path, dtype=dtypes)  # File with TextBlob metrics

    # Step 2: Drop unnecessary columns in the first file
    vader_columns = ['negative', 'neutral', 'positive', 'compound']
    print("Dropping VADER metrics...")
    ddf1 = ddf1.drop(columns=[col for col in vader_columns if col in ddf1.columns])

    # Step 3: Merge the datasets on the 'text' column
    print("Merging datasets...")
    merged_ddf = ddf1.merge(ddf2, on='text', how='inner')

    # Step 4: Select required columns
    selected_columns = [
        'date', 'text', 'textblob_polarity', 'textblob_subjectivity', 'open', 'high',
        'low', 'close', 'volume', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
    ]
    merged_ddf = merged_ddf[selected_columns]

    # Step 5: Save the merged data to CSV
    print("Saving merged data to output file...")
    merged_ddf.to_csv(output_path, single_file=True)
    print(f"Merged file saved to {output_path}")

# Call the function with file paths and output path
merge_bitcoin_tweet_data(
    file1_path='processed_bitcoin_tweets_with_sentiment.csv',
    file2_path='Tweets_and_Bitcoin_Data.csv',
    output_path='bitcoin_dataset_textblob_sentiment.csv'
)
