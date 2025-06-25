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

- `transformers`
- `torch`
- `librosa`
- `pandas`
- `scikit-learn`


## Accuracy Metrics

- **Training Loss Example:**
  | Step | Training Loss |
  |------|---------------|
  | 500 | 1.064000      |

- **Accuracy:**  
  - The model's accuracy on the test set will be reported after evaluation.
- **Other Metrics:**  
  - Precision, recall, and F1-score may be reported if relevant.

---

## 🛠️ How to Use

```bash
# Clone repository and install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Evaluate the model
python evaluate.py
