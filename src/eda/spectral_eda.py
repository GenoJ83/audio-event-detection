"""Acoustic feature analysis: Zero-crossing rate and spectral centroid."""
import numpy as np
from src.data.audio_generator import generate_synthetic_audio, SOUND_CLASSES

def compute_spectral_features(waveform: np.ndarray, sample_rate: int = 16000):
    zcr = ((waveform[:-1] * waveform[1:]) < 0).sum() / len(waveform)
    fft_vals = np.abs(np.fft.rfft(waveform))
    freqs = np.fft.rfftfreq(len(waveform), 1.0 / sample_rate)
    centroid = np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-8)
    return {"zero_crossing_rate": float(round(zcr, 4)), "spectral_centroid_hz": float(round(centroid, 2))}

if __name__ == "__main__":
    for idx, name in enumerate(SOUND_CLASSES):
        w, sr, _ = generate_synthetic_audio(class_idx=idx)
        print(f"[{name}] Stats:", compute_spectral_features(w, sr))
