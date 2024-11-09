#!/usr/bin/env python3

from tensorflow.keras.models import load_model, save_model
import os

# Paths for loading the original model and saving it in .h5 format
original_lstm_model_path = "models/LSTM_best_model.keras"  # Update this to your LSTM model's path
original_gru_model_path = "models/GRU_best_model.keras"  # Update this to your GRU model's path
lstm_h5_model_path = "models/lstm_model.h5"  # Save path for the LSTM model
gru_h5_model_path = "models/gru_model.h5"  # Save path for the GRU model

# Function to load and save models in .h5 format
def convert_and_save_model(original_model_path, h5_model_path, model_name):
    """
    Load a model from a Keras file and save it in .h5 format.

    Parameters:
        original_model_path (str): Path to the original model file.
        h5_model_path (str): Path to save the model in .h5 format.
        model_name (str): The name of the model (LSTM or GRU).
    """
    if not os.path.exists(original_model_path):
        print(f"Error: The original {model_name} model file does not exist at {original_model_path}")
        return

    try:
        # Load the model
        model = load_model(original_model_path)
        # Save the model in .h5 format
        save_model(model, h5_model_path)
        print(f"{model_name} model successfully saved in .h5 format at {h5_model_path}")
    except Exception as e:
        print(f"An error occurred while converting the {model_name} model: {e}")

# Convert and save the LSTM model
convert_and_save_model(original_lstm_model_path, lstm_h5_model_path, "LSTM")

# Convert and save the GRU model
convert_and_save_model(original_gru_model_path, gru_h5_model_path, "GRU")