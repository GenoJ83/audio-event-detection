import pytest
pytest.importorskip("torch")
"""Unit tests for waveform generator, spectrogram, and AudioCNN."""
import torch
from src.data.audio_generator import generate_synthetic_audio
from src.features.spectrogram import extract_log_mel_spectrogram
from src.models.audio_cnn import AudioCNN

def test_audio_generator():
    w, sr, c = generate_synthetic_audio()
    assert len(w) == 32000
    assert sr == 16000

def test_spectrogram_dimensions():
    w, _, _ = generate_synthetic_audio()
    spec = extract_log_mel_spectrogram(w)
    assert spec.shape[0] == 64
    assert spec.shape[1] > 0

def test_audio_cnn_forward():
    model = AudioCNN(num_classes=5)
    x = torch.randn(2, 1, 64, 120)
    out = model(x)
    assert out.shape == (2, 5)
