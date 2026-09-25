"""Random Forest classifier baseline on statistical audio summaries."""
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from src.data.audio_generator import generate_synthetic_audio
from src.features.spectrogram import extract_log_mel_spectrogram

def train_rf_baseline():
    X, y = [], []
    for _ in range(100):
        for c in range(5):
            w, sr, label = generate_synthetic_audio(class_idx=c)
            spec = extract_log_mel_spectrogram(w)
            feats = [spec.mean(), spec.std(), spec.max(), spec.min()]
            X.append(feats)
            y.append(label)

    clf = RandomForestClassifier(n_estimators=30, random_state=42)
    clf.fit(X, y)
    print("Baseline Random Forest Train Accuracy:", clf.score(X, y))
    return clf

if __name__ == "__main__":
    train_rf_baseline()
