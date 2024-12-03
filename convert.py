import h5py
import pandas as pd

def extract_datasets(group, parent_name=""):
    """
    Recursively extract datasets from an HDF5 group.
    Returns a dictionary where keys are dataset names and values are the datasets as NumPy arrays.
    """
    datasets = {}
    for key in group.keys():
        obj = group[key]
        if isinstance(obj, h5py.Group):  # If the object is a group, recurse into it
            datasets.update(extract_datasets(obj, parent_name=f"{parent_name}/{key}"))
        elif isinstance(obj, h5py.Dataset):  # If it's a dataset, add it to the dictionary
            datasets[f"{parent_name}/{key}"] = obj[()]  # Extract as NumPy array
    return datasets

filename = "lstm_sentiment_model.h5"
output_csv = "combined_h5_to_csv_output.csv"

# Open the HDF5 file and extract datasets
with h5py.File(filename, "r") as f:
    # Extract all datasets recursively
    datasets = extract_datasets(f)
    print(f"Found datasets: {list(datasets.keys())}")

    # Combine all datasets into a single DataFrame
    combined_df = pd.DataFrame()
    for dataset_name, data in datasets.items():
        if len(data.shape) == 2:  # If the dataset is 2D
            df = pd.DataFrame(data)
            df["dataset_name"] = dataset_name  # Add a column for dataset name
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        else:
            print(f"Dataset '{dataset_name}' is not 2D (shape: {data.shape}), skipping inclusion.")

    # Save the combined DataFrame to a single CSV file
    combined_df.to_csv(output_csv, index=False)
    print(f"All datasets combined and saved to {output_csv}")
