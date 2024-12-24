import pandas as pd
from textblob import TextBlob
from joblib import Parallel, delayed
import numpy as np

# Redefine necessary functions
def sentiment_scores(sentences):
    """Analyze sentiment using TextBlob for a batch of sentences."""
    results = []
    for sentence in sentences:
        textblob_analysis = TextBlob(str(sentence))
        results.append({
            'textblob_polarity': textblob_analysis.sentiment.polarity,
            'textblob_subjectivity': textblob_analysis.sentiment.subjectivity
        })
    return results

def process_batch_with_window(df, window_size=10):
    """Process a batch of tweets with a sliding window concept for sentiment."""
    sentiments = sentiment_scores(df['text'].tolist())
    sentiment_df = pd.DataFrame(sentiments)
    
    # Initialize the list to store adjusted sentiment scores
    adjusted_polarity = []
    adjusted_subjectivity = []
    
    # Sliding window logic
    for i in range(len(sentiment_df)):
        # Get the last 'window_size' sentiment scores, including the current tweet's sentiment
        window_polarity = sentiment_df['textblob_polarity'][max(0, i - window_size + 1):i + 1]
        window_subjectivity = sentiment_df['textblob_subjectivity'][max(0, i - window_size + 1):i + 1]
        
        # Calculate the average polarity and subjectivity of the window
        adjusted_polarity.append(np.mean(window_polarity))
        adjusted_subjectivity.append(np.mean(window_subjectivity))
    
    # Add adjusted scores to the sentiment dataframe
    sentiment_df['adjusted_polarity'] = adjusted_polarity
    sentiment_df['adjusted_subjectivity'] = adjusted_subjectivity
    
    return pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)

# Reload the dataset
uploaded_file_name = 'updated_dataset_cleaned.csv'  # File in the same folder as the script
df = pd.read_csv(uploaded_file_name, encoding='ISO-8859-1')

# Define batch size and process in batches
batch_size = 10000
batches = [df[i:i + batch_size] for i in range(0, len(df), batch_size)]

# Process the batches in parallel
processed_batches = Parallel(n_jobs=-1)(delayed(process_batch_with_window)(batch) for batch in batches)

# Combine all processed results into one DataFrame
result_df = pd.concat(processed_batches, ignore_index=True)

# Save results to a file
output_file_name = 'updated_processed_bitcoin_tweets_with_sentiment.csv'  # Output file in the same folder
result_df.to_csv(output_file_name, index=False)

# Print confirmation of file path
print(f"Processed file saved as: {output_file_name}")
