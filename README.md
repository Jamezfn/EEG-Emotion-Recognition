# EEG-based Emotion Recognition

This project focuses on recognizing emotions from EEG (Electroencephalography) signals using deep learning models, specifically LSTM and GRU networks. The goal is to train models capable of classifying emotions based on EEG signal data.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup and Installation](#setup-and-installation)
- [Data Preparation](#data-preparation)
- [Model Building and Training](#model-building-and-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [License](#license)
- [Contact Information](#contact-information)
- [Acknowledgments](#acknowledgments)

## Overview

This project employs deep learning techniques to classify emotions from EEG signals. Two models—LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Units)—are used for the classification task. The models have been designed to handle both binary and multi-class emotion classification.

The core components of the system include:

- **Data Preprocessing**: Handles raw EEG data and emotion labels, including filling missing values, scaling features, and preparing data sequences for model input.
- **Model Training**: Builds and trains both LSTM and GRU models.
- **Evaluation**: Assesses the model performance using various metrics.

## Project Structure

```
EEG-Emotion-Recognition/
│
├── data/                          # Folder containing raw and preprocessed data
│   ├── raw_eeg_data.csv           # Raw EEG data (replace with actual data)
│   ├── emotion_labels.json        # Emotion labels corresponding to EEG signals
│   ├── dataset_description.txt    # Dataset description
│   ├── generate_synthetic_data.py # Script for generating synthetic data (optional)
│   └── preprocessed_data/         # Processed data files
│       ├── X_data.npy             # Features after preprocessing
│       └── y_data.npy             # Labels after preprocessing
│
├── models/                        # Folder containing model definitions
│   ├── gru_model.py               # GRU model definition
│   └── lstm_model.py              # LSTM model definition
│
├── scripts/                       # Folder containing executable scripts
│   ├── preprocess_data.py         # Script for loading, preprocessing, and saving data
│   ├── train_model.py             # Script to train and evaluate models (LSTM and GRU)
│   └── evaluate_model.py          # Script to evaluate model performance
│
├── results/                       # Folder for storing training results, logs, and outputs
│   ├── lstm_model_performance.txt # Performance logs for LSTM model
│   └── gru_model_performance.txt  # Performance logs for GRU model
│
└── README.md                      # Project README
```

## Requirements

To run this project, you'll need the following dependencies:

- Python 3.x
- `numpy`
- `pandas`
- `tensorflow` (for model training)
- `scikit-learn`
- `matplotlib` (for plotting results)
- `seaborn` (for advanced plotting)
- `argparse` (for handling command-line arguments)
- `logging` (for logging purposes)

You can install the required libraries by running:

```bash
pip install -r requirements.txt
```

Here is the contents of `requirements.txt`:

```
numpy
pandas
tensorflow
scikit-learn
matplotlib
seaborn
```

Note: `argparse` and `logging` are part of the Python Standard Library and do not need to be installed separately.

## Setup and Installation

1. **Clone this repository** to your local machine:

    ```bash
    git clone https://github.com/Jamezfn/EEG-Emotion-Recognition.git
    cd EEG-Emotion-Recognition
    ```

2. **Install the dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Set up the `data` folder** by placing your raw EEG data (`raw_eeg_data.csv`) and emotion labels (`emotion_labels.json`). Make sure to preprocess the data before training the models.

## Data Preparation

1. **Raw EEG Data**: You need a CSV file containing the EEG signal data. The data should include columns for different EEG features and labels for the corresponding emotions.

2. **Emotion Labels**: Emotion labels are required to train the models. Each row in the raw EEG data must have a corresponding emotion label (e.g., 0 for 'sad', 1 for 'happy', etc.).

3. **Preprocessing**: The `preprocess_data.py` script will clean and preprocess the data:

    - **Load the raw data**: Reads the CSV file containing the EEG data.
    - **Inspect the dataset**: Checks for missing essential columns and provides a summary of the dataset.
    - **Handle missing values**: Fills missing numeric values with the mean and categorical values with the mode.
    - **Scale the features**: Standardizes the EEG signal features.
    - **Create sequences**: Reshapes the data into sequences suitable for LSTM/GRU models.
    - **Save the processed data**: Saves the preprocessed data as `X_data.npy` and `y_data.npy` in the `data/preprocessed_data/` directory.

**Run the preprocessing script like this:**

```bash
python scripts/preprocess_data.py --file_path data/raw_eeg_data.csv --sequence_length 10 --overlap
```

- **Options**:
    - `--file_path`: Path to the raw EEG data CSV file.
    - `--sequence_length`: Length of the sequences for model input (default is 10).
    - `--overlap`: Include this flag to use overlapping sequences.

The processed data will be saved in the `data/preprocessed_data/` folder.

## Model Building and Training

The `train_model.py` script trains the models on the preprocessed data. You can train either the LSTM or GRU model by specifying the `--model_type` argument as "LSTM" or "GRU".

**Train the LSTM model**:

```bash
python -m scripts/train_model --model_type LSTM
```

**Train the GRU model**:

```bash
python -m scripts.train_model --model_type GRU
```

**Options**:

- `--model_type`: Type of model to train (`LSTM` or `GRU`).

**Notes**:

- The script loads `X_data.npy` and `y_data.npy` from the `data/preprocessed_data/` directory.
- The data is split into training and testing sets within the script.
- Training logs and performance metrics are saved in `results/lstm_model_performance.txt` and `results/gru_model_performance.txt`.

## Evaluation

Once the models are trained, the `evaluate_model.py` script can be used to assess their performance. This script computes metrics like accuracy, precision, recall, F1-score, and plots the confusion matrix.

**Convert model from keras to h5**

```bash
python convert_model.py
```

**Evaluate the LSTM model**:

```bash
python scripts/evaluate_model.py --model_path models/lstm_model.h5 --X_test_path data/preprocessed_data/X_test.npy --y_test_path data/preprocessed_data/y_test.npy
```

**Evaluate the GRU model**:

```bash
python scripts/evaluate_model.py --model_path models/gru_model.h5 --X_test_path data/preprocessed_data/X_test.npy --y_test_path data/preprocessed_data/y_test.npy
```

**Options**:

- `--model_path`: Path to the trained model file.
- `--X_test_path`: Path to the test features NumPy file.
- `--y_test_path`: Path to the test labels NumPy file.

The evaluation results, including metrics and confusion matrix plots, will be saved in the `results/` folder.

## Results

The model performance (accuracy, precision, recall, F1-score) will be saved in the `results/` folder. This can be used to compare the performance of the LSTM and GRU models.

## Usage

You can use this project for any EEG-based emotion recognition task by following these steps:

1. **Data Preparation**: Prepare your EEG dataset and ensure it includes the necessary features and labels.

2. **Preprocess the Data**: Use `preprocess_data.py` to clean and preprocess your data.

    ```bash
    python scripts/preprocess_data.py --file_path data/raw_eeg_data.csv
    ```

3. **Train the Models**: Use `train_model.py` to train the LSTM or GRU models.

    ```bash
    python scripts/train_model.py --model_type LSTM
    ```

4. **Evaluate the Models**: Use `evaluate_model.py` to assess model performance and obtain evaluation metrics.

    ```bash
    python scripts/evaluate_model.py --model_path models/lstm_model.h5
    ```

5. **Interpret the Results**: Analyze the results to understand the model's performance and make any necessary adjustments.

## License

This project is licensed under the [MIT License](LICENSE). See the [LICENSE](LICENSE) file for details.

---

## Contact Information

For any questions or contributions, please contact:

- **James**: [email@example.com](mailto:email@example.com)
- **Challo**: [email@example.com](mailto:email@example.com)
- **Fawan**: [email@example.com](mailto:email@example.com)
- **Felicia**: [email@example.com](mailto:email@example.com)
- **Scola**: [email@example.com](mailto:email@example.com)

---

## Acknowledgments

We would like to thank everyone who contributed to this project and the open-source community for providing valuable resources.

---

## Notes

- **Project Structure**:
- **Model Flexibility**: The model definitions in `gru_model.py` and `lstm_model.py` allow for parameter adjustments.
- **Data Splitting**: The data is split into training and testing sets within the `train_model.py` script.
- 

---

Feel free to reach out if you have any questions or need further assistance!
