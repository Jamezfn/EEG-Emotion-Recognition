#!/usr/bin/env python3

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2

def build_lstm_model(input_shape, num_classes=2, dropout_rate=0.2, l2_regularization=0.001):
    """
    Build and compile an LSTM model for binary or multi-class classification.
    
    Parameters:
        input_shape (tuple): The shape of the input data (sequence_length, num_features).
        num_classes (int): Number of classes for classification. Default is 2 for binary classification.
        dropout_rate (float): Dropout rate for regularization. Default is 0.2.
        l2_regularization (float): L2 regularization factor. Default is 0.001.
    
    Returns:
        model: A compiled LSTM model.
    """
    model = Sequential()
    
    # First LSTM layer with dropout and L2 regularization
    model.add(LSTM(
        units=64,
        return_sequences=True,
        input_shape=input_shape,
        kernel_regularizer=l2(l2_regularization)
    ))
    model.add(Dropout(dropout_rate))
    
    # Second LSTM layer
    model.add(LSTM(
        units=32,
        kernel_regularizer=l2(l2_regularization)
    ))
    model.add(Dropout(dropout_rate))
    
    # Output layer
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else:
        model.add(Dense(num_classes, activation='softmax'))
        loss = 'categorical_crossentropy'
    
    # Compile the model
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    
    return model

