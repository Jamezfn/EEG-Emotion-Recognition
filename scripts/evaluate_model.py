#!/usr/bin/env python3
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

def load_trained_model(model_path):
    """
    Load the trained model from a file.
    
    Parameters:
        model_path (str): The path to the trained model file.
    
    Returns:
        keras.models.Sequential: The trained model.
    """
    return load_model(model_path)

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test data.
    
    Parameters:
        model (keras.models.Sequential): The trained model.
        X_test (np.array): Test features.
        y_test (np.array): True labels for the test data.
    
    Returns:
        None: Prints classification report and confusion matrix.
    """
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to class labels

    # Print classification report (precision, recall, F1 score)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    """
    Main execution block for loading a trained model and evaluating it on test data.
    """
    # Replace with the actual file path to your saved model
    model_path = "path_to_saved_model.h5"

    # Load your trained model
    model = load_trained_model(model_path)

    # Load test data (replace with actual test data loading logic)
    X_test = np.load('X_test.npy')  # Test data features
    y_test = np.load('y_test.npy')  # Test data labels

    # Evaluate the model on the test data
    evaluate_model(model, X_test, y_test)

