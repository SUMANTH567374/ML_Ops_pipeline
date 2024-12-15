import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# Step 1: Filter Dataset for Frequent Jobs
def filter_frequent_jobs(data, threshold=50):
    """
    Filters the dataset to include only rows where the job title appears
    more than the specified threshold.
    """
    job_counts = data['Job Title'].value_counts()
    frequent_jobs = job_counts[job_counts > threshold].index
    return data[data['Job Title'].isin(frequent_jobs)]

# Step 2: Encode Categorical Columns
def encode_columns(data, columns_to_encode):
    """
    Encodes specified categorical columns using LabelEncoder.
    Returns the encoded DataFrame and the encoders used.
    """
    label_encoders = {}
    for column in columns_to_encode:
        encoder = LabelEncoder()
        data[column] = encoder.fit_transform(data[column])
        label_encoders[column] = encoder
    return data, label_encoders

# Main Processing Pipeline
def process_data(data, threshold=50, columns_to_encode=None):
    """
    Main function to process the dataset.
    - Filters for frequent jobs.
    - Encodes specified categorical columns.
    """
    # Drop rows with missing values
    data.dropna(inplace=True)

    # Filter for frequent jobs
    filtered_data = filter_frequent_jobs(data, threshold)

    # Encode categorical columns
    if columns_to_encode:
        filtered_data, encoders = encode_columns(filtered_data, columns_to_encode)
    else:
        encoders = None

    return filtered_data, encoders

# Step 3: Process and Save Train and Test Data
def process_and_save(train_path, test_path, output_dir, threshold=50, columns_to_encode=None):
    """
    Processes the train and test datasets and saves the processed versions.
    """
    # Load datasets
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Process datasets
    processed_train, train_encoders = process_data(train_data, threshold, columns_to_encode)
    processed_test, test_encoders = process_data(test_data, threshold, columns_to_encode)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save processed data
    train_output_path = os.path.join(output_dir, "train_processed.csv")
    test_output_path = os.path.join(output_dir, "test_processed.csv")

    processed_train.to_csv(train_output_path, index=False)
    processed_test.to_csv(test_output_path, index=False)

    print("Train data processed and saved to:", train_output_path)
    print("Test data processed and saved to:", test_output_path)

    return train_encoders, test_encoders

# Define paths
train_path = './data/raw/train.csv'
test_path = './data/raw/test.csv'
output_dir = './data/processed'

# Define parameters
columns_to_encode = ['Gender', 'Education Level', 'Job Title']

# Process and save datasets
train_encoders, test_encoders = process_and_save(
    train_path, 
    test_path, 
    output_dir, 
    threshold=50, 
    columns_to_encode=columns_to_encode
)
