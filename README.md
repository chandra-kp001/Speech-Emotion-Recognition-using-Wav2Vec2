# Speech-Emotion-Recognition-using-Wav2Vec2

## Project Description

This project performs emotion classification on speech audio files. The model is trained on a dataset of audio recordings, each labeled with an emotion. The pipeline includes data loading, preprocessing, feature extraction, model training, and evaluation.

## Dataset Overview

The dataset contains audio file paths and corresponding emotion labels. Example:

| Index | audio_paths                                                | emotion |
|-------|------------------------------------------------------------|---------|
| 0     | /kaggle/input/dataset/combined_data/03-01-03-0...          | 0       |
| 1     | /kaggle/input/dataset/combined_data/03-02-04-0...          | 1       |

## Pre-processing Methodology
1. **Data Loading:**  
   - Audio files are loaded from specified paths using libraries such as `librosa` or `torchaudio`.
2. **Data Cleaning:**  
   - there is no missing and correpted audio file.
2. **Label Encoding:**  
   - I extracts the emotion code from an audio filename (like '03-01-05-01-02-01-12.wav'), maps it to a label (e.g., '05' → 'angry'), and adds it as a new column emotion in the DataFrame. It uses the RAVDESS filename format.
3. **Feature Extraction:**  
   - Extract features such as MFCCs, mel spectrograms, or directly use raw waveforms.
4. **Visualization:**  
   - first of all i find out value count of each emotion this will helped me to find that there is imbalance in some emotion then i used stratify during split.
   - I ploted its waveform and spectrogram, and plays the audio for each emotion. It helps visualize and listen to how that emotion sounds.
5. **Train/Validation/Test Split:**  
   - Split the dataset into training, validation, and test sets.

## 🧪 Pipeline Overview

### 1. Dataset Loading
- Loads `.wav` audio files recursively from a dataset directory.
- Extracts emotion labels from file names using the RAVDESS naming convention.

### 2. Preprocessing
- Converts emotion codes to human-readable labels.
- Maps each emotion to a unique integer (`label`).
- Loads audio using `librosa` and resamples it to 16kHz.
- Pads or truncates audio to a fixed length (`2 seconds` or 32000 samples).

### 3. Dataset Preparation
- Implements a custom PyTorch `Dataset` class.
- Uses `Wav2Vec2Processor` to tokenize the audio.
- Returns `input_values` and `labels` for each sample.

### 4. Model Setup
- Loads `facebook/wav2vec2-base` as the base model.
- Adds a classification head for multi-class emotion prediction.
- Supports 8 emotion classes: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised.

### 5. Training
- Splits the dataset into train and test sets (80%/20%).
- Uses Hugging Face `Trainer` with:
  - Evaluation after every epoch
  - Logging and model checkpointing
  - Optimized learning rate, weight decay, and batch sizes

### 6. Evaluation
- Computes metrics: **accuracy**, **precision**, **recall**, and **F1-score**.
- Prints a full classification report using `scikit-learn`.

### 7. Prediction
- Runs inference on the test set.
- Maps numeric predictions back to emotion labels.
- Evaluates performance using predicted vs actual labels.

---

## 📦 Requirements
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - librosa
  - IPython
  - torch
  - torchaudio
  - scikit-learn
  - transformers
  - streamlit
  - soundfile


## Emotion Classification Report

This project involves a machine learning model trained to classify emotions from input data. Below is the evaluation report showing the model's performance on the test dataset.

### Classification Report

| Emotion   | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Angry     | 0.69      | 0.74   | 0.72     | 39      |
| Calm      | 0.69      | 0.69   | 0.69     | 75      |
| Disgust   | 0.84      | 0.83   | 0.83     | 75      |
| Fearful   | 0.89      | 0.77   | 0.83     | 75      |
| Happy     | 0.58      | 0.84   | 0.69     | 38      |
| Neutral   | 0.80      | 0.73   | 0.76     | 75      |
| Sad       | 0.92      | 0.91   | 0.91     | 75      |
| Surprised | 0.86      | 0.82   | 0.84     | 39      |

### Overall Metrics

- **Accuracy**: 0.79  
- **Macro Avg F1-Score**: 0.78  
- **Weighted Avg F1-Score**: 0.79
- **Weighted Avg Precision**: 0.80

## Metrics Explained

- **Precision**: Measures how many of the predicted labels are actually correct.
- **Recall**: Measures how many of the actual labels were correctly predicted.
- **F1-Score**: Harmonic mean of precision and recall. It balances the two metrics.
- **Support**: Number of actual occurrences of each class in the dataset.

## Insights

- The model performs best on the **Sad** (F1-score: 0.91) and **Disgust** (F1-score: 0.83) classes.
- **Happy** and **Calm** emotions show relatively lower F1-scores (0.69), suggesting a need for improvement, possibly via more data, feature tuning, or class balancing.
- Overall model accuracy is **79%**, indicating strong general performance across multiple emotion classes.

---
## 🚀 How to Run the App

### 1️⃣ Install Requirements

Ensure Python 3.7+ is installed. Then install required packages:

pip install -r requirements.txt

---

## 2️⃣ Model Directory

Ensure your trained model is saved at:

This folder must include Hugging Face-compatible files:

- `pytorch_model.bin`
- `config.json`
- `preprocessor_config.json`
- `tokenizer_config.json`
- other related files

To save your model and processor:


``python
model.save_pretrained("results/trained_model")
processor.save_pretrained("results/trained_model")

##3️⃣ Run the App

streamlit run app.py
This will open a browser at:

http://localhost:8501

##4️⃣ Use the Interface
Upload a .wav file (16kHz recommended).

The app will play the audio.

The predicted emotion will be shown on screen.

## demo video Link
https://drive.google.com/file/d/1jM8bubB0YTHdqh6UBhyydXIl2UGdAsJJ/view?usp=sharing
