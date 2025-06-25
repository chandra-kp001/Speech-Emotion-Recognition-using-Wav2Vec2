import os
import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# Fixed emotion mapping
EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_emotion_from_path(path):
    filename = os.path.basename(path)
    parts = filename.split('-')
    emotion_code = parts[2]  # Third segment contains emotion code
    return EMOTION_MAP.get(emotion_code, 'unknown')

# Load dataset
def load_dataset(data_path, max_files=2800):
    paths, emotions = [], []
    for dirname, _, filenames in os.walk(data_path):
        for filename in filenames:
            if len(paths) >= max_files:
                break
            filepath = os.path.join(dirname, filename)
            paths.append(filepath)
            emotions.append(extract_emotion_from_path(filepath))
    return pd.DataFrame({'audio_paths': paths, 'emotion': emotions})

# Create dataset
df = load_dataset('input_file_path')  # Replace with your dataset path

# Create label mappings
emotion_list = sorted(df['emotion'].unique())
EMOTION_TO_INT = {emotion: idx for idx, emotion in enumerate(emotion_list)}
INT_TO_EMOTION = {idx: emotion for emotion, idx in EMOTION_TO_INT.items()}
df['labels'] = df['emotion'].map(EMOTION_TO_INT)

# Dataset class
class SpeechEmotionDataset(Dataset):
    def __init__(self, df, processor, max_length=32000, sampling_rate=16000):
        self.df = df
        self.processor = processor
        self.max_length = max_length
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = self.df.iloc[idx]['audio_paths']
        label = self.df.iloc[idx]['labels']
        
        # Load and process audio
        speech, _ = librosa.load(audio_path, sr=self.sampling_rate)
        if len(speech) > self.max_length:
            speech = speech[:self.max_length]
        else:
            speech = np.pad(speech, (0, self.max_length - len(speech)), 'constant')
        
        inputs = self.processor(
            speech,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        return {
            'input_values': inputs.input_values.squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['labels'], random_state=42)

# Initialize model
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    'facebook/wav2vec2-base',
    num_labels=len(EMOTION_TO_INT)
)

# Create datasets
train_dataset = SpeechEmotionDataset(train_df, processor)
test_dataset = SpeechEmotionDataset(test_df, processor)

# Training setup
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=12,
    weight_decay=0.01,
    logging_dir='./logs',
    report_to=[],
    load_best_model_at_end=True
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Train model
trainer.train()

# Evaluation
results = trainer.evaluate()
print("\nEvaluation Results:")
for metric, value in results.items():
    if metric != 'eval_loss':
        print(f"{metric}: {value:.4f}")

# Generate predictions
predictions = trainer.predict(test_dataset)
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)

# Convert to emotion labels
y_true_emotions = [INT_TO_EMOTION[i] for i in y_true]
y_pred_emotions = [INT_TO_EMOTION[i] for i in y_pred]

# Classification report
print("\nClassification Report:")
print(classification_report(y_true_emotions, y_pred_emotions, target_names=emotion_list))