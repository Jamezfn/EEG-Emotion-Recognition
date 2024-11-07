#!/usr/bin/env python3
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from models.gru_model import build_gru_model
from models.lstm_model import build_lstm_model

def load_preprocessed_data():
    """
    Load the preprocessed EEG data and the emotion labels.
    """
    X = np.load('X_data.npy')
    y = np.load('y_data.npy')
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.

    Parameters:
        X (np.array): Input features (EEG data).
        y (np.array): Labels (emotion labels).
        test_size (float): The proportion of the data to be used as the test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test: The split data.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_type="LSTM"):
    """
    Train and evaluate the chosen model (LSTM or GRU).

    Parameters:
        X_train (np.array): Training data.
        X_test (np.array): Testing data.
        y_train (np.array): Training labels.
        y_test (np.array): Testing labels.
        model_type (str): The model type ('LSTM' or 'GRU').

    Returns:
        float: The accuracy of the model on the test data.
    """
    # Determine model type
    if model_type == "LSTM":
        model = build_lstm_model(X_train.shape[1:])
    elif model_type == "GRU":
        model = build_gru_model(X_train.shape[1:])
    else:
        raise ValueError("Unsupported model type. Choose 'LSTM' or 'GRU'.")

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    return accuracy

if __name__ == "__main__":
    """
    Main execution block to train and evaluate models (LSTM or GRU).
    """
    X, y = load_preprocessed_data()

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training LSTM Model:")
    train_and_evaluate_model(X_train, X_test, y_train, y_test, model_type="LSTM")

    print("Training GRU Model:")
    train_and_evaluate_model(X_train, X_test, y_train, y_test, model_type="GRU")

