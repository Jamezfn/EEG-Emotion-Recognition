#!/usr/bin/env python3

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

def build_gru_model(input_shape):
    """
    Build and compile a GRU model.
    
    Parameters:
        input_shape (tuple): The shape of the input data for the model.
    
    Returns:
        model: A compiled GRU model.
    """
    model = Sequential()
    model.add(GRU(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1, activation='sigmoid'))  # For binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

