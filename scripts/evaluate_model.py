#!/usr/bin/env python3

import numpy as np
import argparse
import os
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve, roc_curve, auc
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer

def load_trained_model(model_path):
    try:
        model = load_model(model_path)
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

def evaluate_model(model, X_test, y_test, class_names=None, output_dir=None):
    if model is None or X_test is None or y_test is None:
        logging.error("Model, X_test, or y_test is None. Exiting evaluation.")
        return

    # Make predictions on the test data
    y_pred_probs = model.predict(X_test)
    
    if y_pred_probs.shape[1] == 1:  # Binary classification
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    else:  # Multi-class classification
        y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Encode y_test if it's in string format
    if isinstance(y_test.flatten()[0], str):
        le = LabelEncoder()
        y_test = le.fit_transform(y_test)  # Encode y_test to integers
    y_test_labels = y_test if y_test.ndim == 1 else np.argmax(y_test, axis=1)

    # Generate class names if not provided and if integer labels are used
    if class_names is None:
        class_names = le.classes_ if isinstance(y_test.flatten()[0], np.integer) else [str(i) for i in range(len(np.unique(y_test_labels)))]

    # Calculate accuracy, report, and confusion matrix
    accuracy = accuracy_score(y_test_labels, y_pred)
    report = classification_report(y_test_labels, y_pred, labels=np.arange(len(class_names)), digits=4)
    cm = confusion_matrix(y_test_labels, y_pred)

    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info("Classification Report:")
    logging.info(f"\n{report}")
    logging.info("Confusion Matrix:")
    logging.info(f"\n{cm}")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        metrics_file = os.path.join(output_dir, 'evaluation_metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report + "\n")
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(cm) + "\n")
        logging.info(f"Evaluation metrics saved to {metrics_file}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    if output_dir:
        plot_file = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(plot_file)
        logging.info(f"Confusion matrix plot saved to {plot_file}")
    else:
        plt.show()

    # Multi-class Precision-Recall curve (for multi-class classification)
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test_labels)
    y_pred_bin = lb.transform(y_pred)

    precision, recall, _ = precision_recall_curve(y_test_bin.ravel(), y_pred_bin.ravel())
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Multi-class)')
    if output_dir:
        pr_curve_file = os.path.join(output_dir, 'precision_recall_curve.png')
        plt.savefig(pr_curve_file)
        logging.info(f"Precision-Recall curve saved to {pr_curve_file}")
    else:
        plt.show()

    # Optional: Multi-class ROC curve and AUC
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(y_pred_bin.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i in range(y_pred_bin.shape[1]):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve (Multi-class)')
    plt.legend(loc="lower right")
    if output_dir:
        roc_curve_file = os.path.join(output_dir, 'roc_curve.png')
        plt.savefig(roc_curve_file)
        logging.info(f"ROC curve saved to {roc_curve_file}")
    else:
        plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='Evaluate a trained model on test data.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--X_test_path', type=str, required=True, help='Path to the test features file (numpy array).')
    parser.add_argument('--y_test_path', type=str, required=True, help='Path to the test labels file (numpy array).')
    parser.add_argument('--class_names', type=str, nargs='*', default=None, help='List of class names.')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save evaluation results.')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        logger.error(f"Model file does not exist: {args.model_path}")
        exit(1)
    if not os.path.exists(args.X_test_path):
        logger.error(f"Test features file does not exist: {args.X_test_path}")
        exit(1)
    if not os.path.exists(args.y_test_path):
        logger.error(f"Test labels file does not exist: {args.y_test_path}")
        exit(1)

    model = load_trained_model(args.model_path)

    X_test = np.load(args.X_test_path)
    y_test = np.load(args.y_test_path)

    evaluate_model(model, X_test, y_test, class_names=args.class_names, output_dir=args.output_dir)
