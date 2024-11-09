import logging
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import argparse

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        data.columns = data.columns.str.replace('# ', '')  # Clean column names
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def check_dataset(data, required_columns=None):
    """
    Inspect the dataset to understand its structure and ensure necessary columns are present.

    Parameters:
        data (pd.DataFrame): The loaded EEG dataset.
        required_columns (list): List of required column names.

    Raises:
        ValueError: If essential columns are missing.

    Prints:
        The first few rows, info, and summary statistics of the dataset.
    """
    if required_columns is None:
        required_columns = ['mean_0_a', 'mean_1_a', 'mean_2_a', 'mean_3_a', 'label']  # Replace with actual column names if different

    print("First few rows of the dataset:")
    print("Columns in the dataset:")
    print(data.columns)
    print(data.head())

    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        logging.error(f"Missing essential columns: {missing_columns}")
        raise ValueError(f"Dataset is missing essential columns: {missing_columns}")
    else:
        logging.info(f"All essential columns are present: {required_columns}")

    print("\nDataset Summary:")
    print(data.info())
    print("\nDataset Description:")
    print(data.describe())

def handle_missing_values(data):
    # Separate the label column from the numeric data
    label_column = data['label']
    data = data.drop(columns=['label'])
    
    # Fill missing values with the mean of each column (ignoring non-numeric columns)
    data_filled = data.fillna(data.mean())
    
    # Add the label column back
    data_filled['label'] = label_column
    
    # Alternatively, you could fill missing labels with the mode
    # data_filled['label'] = data_filled['label'].fillna(data_filled['label'].mode()[0])
    
    return data_filled

def create_sequences(data, sequence_length, overlap):
    sequences = []
    labels = []
    step = sequence_length - overlap
    for i in range(0, len(data) - sequence_length + 1, step):
        sequence = data.iloc[i:i + sequence_length, :-1].values  # Exclude the label column
        label = data.iloc[i + sequence_length - 1, -1]  # The last label in the sequence
        sequences.append(sequence)
        labels.append(label)
    return np.array(sequences), np.array(labels)


def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    # Split the data into train and temporary (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, random_state=random_state)
    
    # Split the temporary data into validation and test sets
    val_size_adjusted = val_size / (test_size + val_size)  # Adjust the validation size relative to the temporary split
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_data(file_path, sequence_length=10, overlap=True, random_seed=42):
    data = load_data(file_path)
    if data is None:
        raise Exception("Failed to load data.")

    # Check dataset structure and contents
    check_dataset(data)

    # Handle missing values
    data = handle_missing_values(data)

    # Create sequences for LSTM/GRU
    X, y = create_sequences(data, sequence_length, overlap)

    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, random_state=random_seed)

    # Scale the data
    eeg_feature_shape = X_train.shape[2]
    scaler = StandardScaler()

    # Reshaping and scaling for LSTM
    X_train = X_train.reshape(-1, eeg_feature_shape)
    X_train = scaler.fit_transform(X_train).reshape(-1, sequence_length, eeg_feature_shape)

    X_val = X_val.reshape(-1, eeg_feature_shape)
    X_val = scaler.transform(X_val).reshape(-1, sequence_length, eeg_feature_shape)

    X_test = X_test.reshape(-1, eeg_feature_shape)
    X_test = scaler.transform(X_test).reshape(-1, sequence_length, eeg_feature_shape)

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(description="Preprocess EEG data for emotion recognition.")
    parser.add_argument('--file_path', type=str, required=True, help="Path to the EEG data CSV file.")
    parser.add_argument('--sequence_length', type=int, default=10, help="Length of the sequences for model input.")
    parser.add_argument('--overlap', action='store_true', help="Use overlapping sequences.")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for data splitting.")
    parser.add_argument('--output_dir', type=str, default='data/preprocessed_data/', help="Directory to save preprocessed data.")
    args = parser.parse_args()

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        file_path=args.file_path,
        sequence_length=args.sequence_length,
        overlap=args.overlap,
        random_seed=args.random_seed
    )

    os.makedirs(args.output_dir, exist_ok=True)

    np.save(os.path.join(args.output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(args.output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(args.output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(args.output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(args.output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(args.output_dir, 'y_test.npy'), y_test)

    logging.info(f"Preprocessed data saved in {args.output_dir}")
    logging.info(f"X_train shape: {X_train.shape}")
    logging.info(f"X_val shape: {X_val.shape}")
    logging.info(f"X_test shape: {X_test.shape}")
    logging.info(f"y_train shape: {y_train.shape}")
    logging.info(f"y_val shape: {y_val.shape}")
    logging.info(f"y_test shape: {y_test.shape}")
