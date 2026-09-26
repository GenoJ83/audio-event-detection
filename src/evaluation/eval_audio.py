"""Acoustic event classification metrics and Top-K accuracy."""
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from src.data.audio_generator import SOUND_CLASSES

def evaluate_predictions(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    rep = classification_report(y_true, y_pred, target_names=SOUND_CLASSES, zero_division=0)
    print(f"Top-1 Accuracy: {acc:.2%}")
    return {"accuracy": float(round(acc, 4)), "report": rep}
