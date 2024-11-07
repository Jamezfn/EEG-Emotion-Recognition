# EEG-based Emotion Recognition

This project focuses on recognizing emotions from EEG (Electroencephalography) signals using machine learning models, specifically LSTM and GRU networks. The goal is to train deep learning models to classify emotions based on EEG signal data.

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

## Overview

This project employs deep learning techniques to classify emotions from EEG signals. Two models—LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Units)—are used for the classification task. The EEG signals are preprocessed, then split into training and test sets for evaluation. 

The core components of the system include:
- **Data Preprocessing**: Handles raw EEG data and emotion labels, filling missing values, and scaling features.
- **Model Training**: Builds and trains both LSTM and GRU models.
- **Evaluation**: Assesses the model performance using accuracy metrics.

## Project Structure

```
EEG-Emotion-Recognition/
│
├── data/                        # Folder containing raw and preprocessed data
│   ├── raw_eeg_data.csv         # Raw EEG data (replace with actual data)
│   ├── emotion_labels.json      # Emotion labels corresponding to EEG signals
│   ├── dataset_description.txt # Dataset description
│   ├── generate_synthetic_data.py  # Script for generating synthetic data (optional)
│   └── preprocessed_data/       # Processed data files
│       ├── X_data.npy           # Features after preprocessing
│       └── y_data.npy           # Labels after preprocessing
│
├── models/                      # Folder containing model definitions
│   ├── gru_model.py             # GRU model definition
│   └── lstm_model.py            # LSTM model definition
│
├── results/                     # Folder for storing training results, logs, and outputs
│   ├── lstm_model_performance.txt # Performance logs for LSTM model
│   └── gru_model_performance.txt  # Performance logs for GRU model
│
├── preprocess_data.py           # Script for loading, preprocessing, and saving data
├── train_model.py               # Script to train and evaluate models (LSTM and GRU)
├── evaluate_model.py            # Script to evaluate model performance
└── README.md                    # Project README
```

## Requirements

To run this project, you'll need the following dependencies:

- Python 3.x
- `numpy`
- `pandas`
- `tensorflow` (for model training)
- `scikit-learn`
- `matplotlib` (for plotting results)
- `keras`

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
keras
```

## Setup and Installation

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/Jamezfn/EEG-Emotion-Recognition.git
    cd EEG-Emotion-Recognition
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up the `data` folder by placing your raw EEG data (`raw_eeg_data.csv`) and emotion labels (`emotion_labels.json`). Make sure to preprocess the data before training the models.

## Data Preparation

1. **Raw EEG Data**: You need a CSV file containing the EEG signal data. The data should include columns for different features (EEG signal readings) and labels for the corresponding emotions.

2. **Emotion Labels**: Emotion labels are required to train the models. Each row in the raw EEG data must have a corresponding emotion label (e.g., 0 for 'sad', 1 for 'happy').

3. **Preprocessing**: The `preprocess_data.py` script will clean and preprocess the data:
    - Load the raw data
    - Handle missing values
    - Scale the features
    - Split the data into training and testing sets
    - Save the processed data as `.npy` files (`X_data.npy` and `y_data.npy`)

Run the preprocessing script like this:

```bash
python preprocess_data.py
```

The processed data will be saved in the `data/preprocessed_data/` folder.

## Model Building and Training

The `train_model.py` script trains the models on the preprocessed data. You can train either the LSTM or GRU model by passing the `model_type` argument as "LSTM" or "GRU".

1. **Train the LSTM model**:

    ```bash
    python train_model.py --model_type LSTM
    ```

2. **Train the GRU model**:

    ```bash
    python train_model.py --model_type GRU
    ```

Both models will be trained on the data and evaluated on the test set. The performance will be logged in `results/lstm_model_performance.txt` and `results/gru_model_performance.txt`.

## Evaluation

Once the models are trained, the `evaluate_model.py` script can be used to assess their performance. This script computes metrics like accuracy and can visualize the results.

To evaluate a trained model, run:

```bash
python evaluate_model.py --model_type LSTM
```

or

```bash
python evaluate_model.py --model_type GRU
```

The evaluation results will be displayed on the terminal and logged in the `results/` folder.

## Results

The model performance (accuracy) will be saved in the `results/` folder. This can be used to compare the performance of the LSTM and GRU models.

## Usage

You can use this project for any EEG-based emotion recognition task by following these steps:

1. **Preprocess the data**: Ensure the data is clean and ready for training.
2. **Train the models**: Use LSTM or GRU models for training.
3. **Evaluate the models**: Assess the model performance using evaluation scripts.
4. **Visualize results**: Use any of the performance metrics to improve the model.

## License

This project is licensed under the [Challo, James, Fawan, Felicia, Scola] - see the [LICENSE](LICENSE) file for details.

---

### Notes:
 INCOMing
