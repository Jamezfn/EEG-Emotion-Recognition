#!/usr/bin/env python3

import numpy as np
import argparse
import os
import logging
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pandas as pd

def load_trained_model(model_path):
    """
    Load the trained model from a file.

    Parameters:
        model_path (str): The path to the trained model file.

    Returns:
        keras.models.Model: The trained model.
    """
    try:
        model = load_model(model_path)
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

def evaluate_model(model, X_test, y_test, class_names=None, output_dir=None):
    """
    Evaluate the trained model on the test data.

    Parameters:
        model (keras.models.Model): The trained model.
        X_test (np.array): Test features.
        y_test (np.array): True labels for the test data.
        class_names (list): List of class names for labeling.
        output_dir (str): Directory to save evaluation results.

    Returns:
        None: Prints classification report and confusion matrix.
    """
    if model is None or X_test is None or y_test is None:
        logging.error("Model, X_test, or y_test is None. Exiting evaluation.")
        return

    # Make predictions on the test data
    y_pred_probs = model.predict(X_test)
    
    # Determine if it's binary or multi-class classification
    if y_pred_probs.shape[1] == 1:
        # Binary classification
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    else:
        # Multi-class classification
        y_pred = np.argmax(y_pred_probs, axis=1)
        
    # Convert y_test to integer labels if it's one-hot encoded
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test_labels = np.argmax(y_test, axis=1)
    else:
        y_test_labels = y_test.flatten().astype(int)
        
    # Check if class_names were provided or create default ones
    if class_names is None:
        logging.warning("Class names not provided. Using default class names.")
        class_names = [f'Class{i}' for i in range(len(np.unique(y_test_labels)))]
    
    # Ensure that the number of class names matches the number of unique labels
    if len(class_names) != len(np.unique(y_test_labels)):
        logging.error("Mismatch between class_names and unique labels in the dataset!")
        raise ValueError("The length of class_names must match the number of unique labels in y_test.")
    
    # Compute metrics
    accuracy = accuracy_score(y_test_labels, y_pred)
    report = classification_report(y_test_labels, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_test_labels, y_pred)

    # Log metrics
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info("Classification Report:")
    logging.info(f"\n{report}")
    logging.info("Confusion Matrix:")
    logging.info(f"\n{cm}")

    # Save metrics to a file
    if output_dir:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save evaluation metrics
        metrics_file = os.path.join(output_dir, 'evaluation_metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report + "\n")
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(cm) + "\n")
            f.flush()  # Ensuring the results are written to file
        logging.info(f"Evaluation metrics saved to {metrics_file}")
        
        # Save confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plot_file = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(plot_file)
        logging.info(f"Confusion matrix plot saved to {plot_file}")
        
        # Save accuracy plot
        plt.figure()
        plt.plot(model.history.history['accuracy'], label='Accuracy')
        plt.plot(model.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        accuracy_plot_file = os.path.join(output_dir, 'accuracy_plot.png')
        plt.savefig(accuracy_plot_file)
        logging.info(f"Accuracy plot saved to {accuracy_plot_file}")

        # Save loss curve plot
        plt.figure()
        plt.plot(model.history.history['loss'], label='Loss')
        plt.plot(model.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        loss_curve_file = os.path.join(output_dir, 'loss_curve.png')
        plt.savefig(loss_curve_file)
        logging.info(f"Loss curve plot saved to {loss_curve_file}")

        # Save model comparison
        model_comparison = [
            ['Model', 'Accuracy'],
            ['LSTM', accuracy]  # Change to include LSTM/GRU comparison results
        ]
        comparison_file = os.path.join(output_dir, 'model_comparison.csv')
        with open(comparison_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(model_comparison)
        logging.info(f"Model comparison saved to {comparison_file}")

        # Save predictions
        predictions_df = pd.DataFrame({
            'True Label': y_test_labels,
            'Predicted Label': y_pred
        })
        predictions_file = os.path.join(output_dir, 'predictions.csv')
        predictions_df.to_csv(predictions_file, index=False)
        logging.info(f"Predictions saved to {predictions_file}")

    else:
        plt.show()

if __name__ == "__main__":
    """
    Main execution block for loading a trained model and evaluating it on test data.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate a trained model on test data.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--X_test_path', type=str, required=True, help='Path to the test features file (numpy array).')
    parser.add_argument('--y_test_path', type=str, required=True, help='Path to the test labels file (numpy array).')
    parser.add_argument('--class_names', type=str, nargs='*', default=None, help='List of class names.')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save evaluation results.')
    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.model_path):
        logger.error(f"Model file does not exist: {args.model_path}")
        exit(1)
    if not os.path.exists(args.X_test_path):
        logger.error(f"Test features file does not exist: {args.X_test_path}")
        exit(1)
    if not os.path.exists(args.y_test_path):
        logger.error(f"Test labels file does not exist: {args.y_test_path}")
        exit(1)

    # Load your trained model
    model = load_trained_model(args.model_path)

    # Load test data
    X_test = np.load(args.X_test_path)
    y_test = np.load(args.y_test_path)

    # Evaluate the model on the test data
    evaluate_model(model, X_test, y_test, class_names=args.class_names, output_dir=args.output_dir)
