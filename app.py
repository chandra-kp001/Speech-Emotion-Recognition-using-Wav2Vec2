# app.py

import streamlit as st
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

# Load saved model and processor
MODEL_PATH = "./results/train_model"
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
model.eval()

# Your label mapping must match training order
INT_TO_EMOTION = {
    0: 'angry',
    1: 'calm',
    2: 'disgust',
    3: 'fearful',
    4: 'happy',
    5: 'neutral',
    6: 'sad',
    7: 'surprised'
}

# UI
st.set_page_config(page_title="Speech Emotion Recognition")
st.title("🎙️ Speech Emotion Recognition")
st.markdown("Upload a `.wav` audio file to classify the emotion.")

# File uploader
uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Load audio
    y, sr = librosa.load(uploaded_file, sr=16000)

    # Pad or trim to 2 seconds
    max_len = 32000
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)))

    # Tokenize and predict
    inputs = processor(y, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_id = torch.argmax(logits, dim=-1).item()

    predicted_emotion = INT_TO_EMOTION[predicted_id]
    st.success(f"**Predicted Emotion: {predicted_emotion.capitalize()}**")
