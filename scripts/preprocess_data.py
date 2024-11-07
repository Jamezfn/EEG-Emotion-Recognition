#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_data(file_path):
    """
    Load the EEG dataset from a CSV file.
    
    Parameters:
        file_path (str): Path to the CSV file containing EEG data.
        
    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    return pd.read_csv(file_path)

def check_dataset(data):
    """
    Inspect the dataset to understand its structure and ensure necessary columns are present.
    
    Parameters:
        data (pd.DataFrame): The loaded EEG dataset.
        
    Prints:
        The first few rows, info, and summary statistics of the dataset.
    """
    print("First few rows of the dataset:")
    print(data.head())
    
    required_columns = ['EEG Signal', 'Emotion Label']  # Replace with actual column names if different
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        print("Warning: Missing essential columns:", missing_columns)
    else:
        print("All essential columns are present:", required_columns)
    
    print("\nDataset Summary:")
    print(data.info())
    print("\nDataset Description:")
    print(data.describe())

def handle_missing_values(data):
    """
    Handle missing values in the EEG dataset by filling them with the mean.
    
    Parameters:
        data (pd.DataFrame): The EEG dataset.
        
    Returns:
        pd.DataFrame: Dataset with missing values filled.
    """
    imputer = SimpleImputer(strategy="mean")
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    return data_imputed

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
    eeg_columns = [col for col in data.columns if 'EEG' in col]
    data[eeg_columns] = scaler.fit_transform(data[eeg_columns])
    return data

def create_sequences(data, sequence_length=10):
    """
    Reshapes EEG data into sequences for LSTM/GRU models.
    
    Parameters:
        data (pd.DataFrame): The preprocessed EEG dataset.
        sequence_length (int): The number of time steps per sequence.
        
    Returns:
        X (np.array): 3D array of input data sequences.
        y (np.array): Labels for each sequence.
    """
    eeg_columns = [col for col in data.columns if 'EEG' in col]  # Adjust if necessary
    sequences = []
    labels = []

    for i in range(len(data) - sequence_length + 1):
        eeg_sequence = data[eeg_columns].iloc[i:i + sequence_length].values
        label = data['Emotion Label'].iloc[i + sequence_length - 1]
        sequences.append(eeg_sequence)
        labels.append(label)

    X = np.array(sequences)
    y = np.array(labels)

    return X, y

def preprocess_data(file_path):
    """
    Load, inspect, clean, and preprocess the EEG dataset, and reshape it into sequences for LSTM/GRU.
    
    Parameters:
        file_path (str): Path to the CSV dataset file.
        
    Returns:
        X (np.array): Preprocessed and reshaped EEG data.
        y (np.array): Corresponding emotion labels.
    """
    # Load the data
    data = load_data(file_path)
    
    # Check dataset structure and contents
    check_dataset(data)
    
    # Handle missing values
    data = handle_missing_values(data)
    
    # Scale EEG signal columns
    data = scale_data(data)
    
    # Create sequences for LSTM/GRU
    X, y = create_sequences(data)
    
    return X, y

if __name__ == "__main__":
    """
    Main execution block for loading, cleaning, and preprocessing the dataset.
    Adjust the file path to point to your CSV file.
    """
    # Example file path; replace with your actual file location
    file_path = "data/eeg_data.csv"
    
    # Preprocess the data
    X, y = preprocess_data(file_path)
    
    print("\nPreprocessed Data Sample (X):")
    print(X.shape)  # Print shape of the processed data
    print("\nSample Labels (y):")
    print(y[:5])  # Print first 5 labels

