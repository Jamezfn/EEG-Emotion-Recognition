#!/usr/bin/env python3

import numpy as np
import argparse
import logging
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder  # Ensure this is imported

from models.gru_model import build_gru_model
from models.lstm_model import build_lstm_model

def load_preprocessed_data(data_dir):
    """
    Load the preprocessed EEG data and the emotion labels.

    Parameters:
        data_dir (str): Directory where the preprocessed data is stored.

    Returns:
        Tuple of NumPy arrays: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_labels(y_train, y_val, y_test, num_classes):
    """
    Preprocess the labels by converting them to numeric values and one-hot encoding if necessary.

    Parameters:
        y_train, y_val, y_test: The label arrays for training, validation, and testing.
        num_classes: The number of classes (for one-hot encoding).

    Returns:
        Tuple of the processed labels (y_train, y_val, y_test).
    """
    label_encoder = LabelEncoder()

    # Convert string labels to numeric
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)

    # One-hot encode if more than 2 classes
    if num_classes > 2:
        y_train = to_categorical(y_train, num_classes=num_classes)
        y_val = to_categorical(y_val, num_classes=num_classes)
        y_test = to_categorical(y_test, num_classes=num_classes)

    return y_train, y_val, y_test

def train_and_evaluate_model(X_train, X_val, X_test, y_train, y_val, y_test,
                             model_type="LSTM", epochs=50, batch_size=32,
                             output_dir='models/', num_classes=2):
    """
    Train and evaluate the chosen model (LSTM or GRU).

    Parameters:
        X_train (np.array): Training data.
        X_val (np.array): Validation data.
        X_test (np.array): Testing data.
        y_train (np.array): Training labels.
        y_val (np.array): Validation labels.
        y_test (np.array): Testing labels.
        model_type (str): The model type ('LSTM' or 'GRU').
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        output_dir (str): Directory to save the trained model.
        num_classes (int): Number of classes in the labels.

    Returns:
        float: The accuracy of the model on the test data.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Build the model
    input_shape = X_train.shape[1:]
    if model_type == "LSTM":
        model = build_lstm_model(input_shape, num_classes=num_classes)
    elif model_type == "GRU":
        model = build_gru_model(input_shape, num_classes=num_classes)
    else:
        raise ValueError("Unsupported model type. Choose 'LSTM' or 'GRU'.")

    # Callbacks
    checkpoint_path = os.path.join(output_dir, f'{model_type}_best_model.keras')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, verbose=1),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    # Load the best model
    model.load_weights(checkpoint_path)

    # Evaluate on test data
    y_pred_probs = model.predict(X_test)
    if num_classes == 2:
        # Binary classification
        y_pred = (y_pred_probs > 0.5).astype(int)
    else:
        # Multi-class classification
        y_pred = np.argmax(y_pred_probs, axis=1)

    # Ensure y_test is in the same format
    if num_classes == 2:
        y_test = (y_test > 0.5).astype(int)  # Convert binary labels
    else:
        y_test = np.argmax(y_test, axis=1)  # Convert one-hot labels to class indices   

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Log the training history
    history_dict = history.history
    logging.info(f"Training Loss: {history_dict['loss'][-1]}")
    logging.info(f"Validation Loss: {history_dict['val_loss'][-1]}")
    logging.info(f"Training Accuracy: {history_dict['accuracy'][-1]}")
    logging.info(f"Validation Accuracy: {history_dict['val_accuracy'][-1]}")

    # Save the final model
    model_save_path = os.path.join(output_dir, f'{model_type}_final_model.keras')
    model.save(model_save_path)
    logging.info(f"Final model saved at {model_save_path}")

    return accuracy

if __name__ == "__main__":
    """
    Main execution block to train and evaluate models (LSTM or GRU).
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Command-line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate LSTM or GRU model.')
    parser.add_argument('--data_dir', type=str, default='data/preprocessed_data/', help='Directory of preprocessed data.')
    parser.add_argument('--model_type', type=str, choices=['LSTM', 'GRU'], default='LSTM', help='Type of model to train.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--output_dir', type=str, default='models/', help='Directory to save the trained model.')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes.')
    args = parser.parse_args()

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data(args.data_dir)

    # Preprocess labels (binary or multi-class)
    y_train, y_val, y_test = preprocess_labels(y_train, y_val, y_test, args.num_classes)

    # Train and evaluate the model
    accuracy = train_and_evaluate_model(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        num_classes=args.num_classes
    )

    logging.info(f"Final Test Accuracy for {args.model_type} model: {accuracy * 100:.2f}%")