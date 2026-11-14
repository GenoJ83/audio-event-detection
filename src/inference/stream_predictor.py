"""Streaming audio buffer event detection pipeline."""
import torch
import numpy as np
from src.data.audio_generator import SOUND_CLASSES
from src.features.spectrogram import extract_log_mel_spectrogram
from src.models.audio_cnn import AudioCNN

class AudioEventDetector:
    def __init__(self, model: AudioCNN = None):
        self.model = model or AudioCNN(num_classes=len(SOUND_CLASSES))
        self.model.eval()

    def detect_event(self, audio_buffer: np.ndarray):
        spec = extract_log_mel_spectrogram(audio_buffer)
        tensor = torch.tensor(spec).unsqueeze(0).unsqueeze(0) # (1, 1, mels, frames)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].numpy()
            pred = int(probs.argmax())

        return {
            "event": SOUND_CLASSES[pred],
            "confidence": float(round(probs[pred], 4)),
            "all_classes": {SOUND_CLASSES[i]: float(round(probs[i], 4)) for i in range(len(SOUND_CLASSES))}
        }
