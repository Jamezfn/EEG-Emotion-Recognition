#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
import logging
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load the EEG dataset from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file containing EEG data.

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame, or None if loading fails.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        logging.error(f"Error: The file {file_path} does not exist.")
        return None
    except pd.errors.EmptyDataError:
        logging.error("Error: The file is empty.")
        return None
    except Exception as e:
        logging.error(f"An error occurred while loading the data: {e}")
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
<<<<<<< HEAD
        required_columns = ['mean_0_a', 'mean_1_a', 'mean_2_a_', 'label']  # Replace with actual column names if different
=======
        required_columns = ['mean_0_a', 'mean_1_a','mean_2_a', 'mean_3_a', 'mean_4_a','mean_d_0_a','mean_d_1_a','mean_d_2_a','mean_d_3_a','mean_d_4_a','mean_d_0_a2','mean_d_1_a2','mean_d_2_a2','mean_d_3_a2','mean_d_4_a2','mean_d_5_a','mean_d_6_a','mean_d_7_a','mean_d_8_a','mean_d_9_a','mean_d_10_a', 'Emotion Label']  # Replace with actual column names if different
>>>>>>> e97bf3f87304656772abbe691a6b43b6a92a9327

    print("First few rows of the dataset:")
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
    """
    Handle missing values in the EEG dataset by filling them with the mean for numeric columns.

    Parameters:
        data (pd.DataFrame): The EEG dataset.

    Returns:
        pd.DataFrame: Dataset with missing values filled.
    """
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy="mean")
    data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

    # Handle non-numeric columns if necessary
    # For example, fill categorical columns with the mode
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            mode = data[col].mode()[0]
            data[col].fillna(mode, inplace=True)

    return data

def scale_data(data):
    """
    Scale EEG signal data using standardization.

    Parameters:
        data (pd.DataFrame): The EEG dataset with signals to be scaled.

    Returns:
        pd.DataFrame: Scaled EEG signal data.
    """
    scaler = StandardScaler()

    # Adjust 'EEG Signal' to match actual EEG signal columns if different
    eeg_columns = [col for col in data.columns if 'mean_' in col]  # Adjust if necessary

    data[eeg_columns] = scaler.fit_transform(data[eeg_columns])

    return data

def create_sequences(data, sequence_length=10, overlap=True):
    """
    Reshapes EEG data into sequences for LSTM/GRU models.

    Parameters:
        data (pd.DataFrame): The preprocessed EEG dataset.
        sequence_length (int): The number of time steps per sequence.
        overlap (bool): Whether sequences should overlap.

    Returns:
        X (np.array): 3D array of input data sequences.
        y (np.array): Labels for each sequence.
    """
    eeg_columns = [col for col in data.columns if 'mean_' in col]  # Adjust if necessary
    sequences = []
    labels = []

    if overlap:
        step = 1
    else:
        step = sequence_length

    for i in range(0, len(data) - sequence_length + 1, step):
        eeg_sequence = data[eeg_columns].iloc[i:i + sequence_length].values
        label = data['mean_1_a'].iloc[i + sequence_length - 1]
        sequences.append(eeg_sequence)
        labels.append(label)

    X = np.array(sequences)
    y = np.array(labels)

    return X, y

def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split the data into training, validation, and test sets.

    Parameters:
        X (np.array): Input features.
        y (np.array): Labels.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random seed.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First, split off the test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True)

    # Then, split the remaining data into training and validation sets
    val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size proportionally
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, random_state=random_state, shuffle=True)

    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_data(file_path, sequence_length=10, overlap=True):
    """
    Load, inspect, clean, and preprocess the EEG dataset, and reshape it into sequences for LSTM/GRU.

    Parameters:
        file_path (str): Path to the CSV dataset file.
        sequence_length (int): The number of time steps per sequence.
        overlap (bool): Whether sequences should overlap.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Load the data
    data = load_data(file_path)
    if data is None:
        raise Exception("Data loading failed.")

    # Check dataset structure and contents
    check_dataset(data)

    # Handle missing values
    data = handle_missing_values(data)

    # Create sequences for LSTM/GRU
    X, y = create_sequences(data, sequence_length=sequence_length, overlap=overlap)

    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Scale EEG signal columns using training data scaler to avoid data leakage
    eeg_feature_shape = X_train.shape[2]  # Number of EEG features

    scaler = StandardScaler()
    X_train = X_train.reshape(-1, eeg_feature_shape)
    X_train = scaler.fit_transform(X_train)
    X_train = X_train.reshape(-1, sequence_length, eeg_feature_shape)

    X_val = X_val.reshape(-1, eeg_feature_shape)
    X_val = scaler.transform(X_val)
    X_val = X_val.reshape(-1, sequence_length, eeg_feature_shape)

    X_test = X_test.reshape(-1, eeg_feature_shape)
    X_test = scaler.transform(X_test)
    X_test = X_test.reshape(-1, sequence_length, eeg_feature_shape)

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    """
    Main execution block for loading, cleaning, and preprocessing the dataset.
    Adjust the file path to point to your CSV file.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Command-line arguments
    parser = argparse.ArgumentParser(description='Preprocess EEG data for emotion recognition.')
    parser.add_argument('--file_path', type=str, default='data/raw_eeg_data.csv', help='Path to the EEG data CSV file.')
    parser.add_argument('--sequence_length', type=int, default=10, help='Length of the sequences for model input.')
    parser.add_argument('--overlap', action='store_true', help='Use overlapping sequences.')
    parser.add_argument('--output_dir', type=str, default='data/preprocessed_data/', help='Directory to save preprocessed data.')
    args = parser.parse_args()

    # Preprocess the data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        file_path=args.file_path,
        sequence_length=args.sequence_length,
        overlap=args.overlap
    )

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Save the processed data
    np.save(os.path.join(args.output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(args.output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(args.output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(args.output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(args.output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(args.output_dir, 'y_test.npy'), y_test)

    logger.info(f"Preprocessed data saved in {args.output_dir}")
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_val shape: {X_val.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"y_val shape: {y_val.shape}")
    logger.info(f"y_test shape: {y_test.shape}")

