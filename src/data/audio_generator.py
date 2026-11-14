"""Synthetic multi-class acoustic waveform simulator."""
import numpy as np

SOUND_CLASSES = ["Siren", "Dog_Bark", "Drilling", "Gunshot", "Street_Music"]

def generate_synthetic_audio(duration_sec: float = 2.0, sample_rate: int = 16000, class_idx: int = 0):
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    rng = np.random.default_rng()
    if class_idx == 0: # Siren: modulating frequency
        freq = 600 + 300 * np.sin(2 * np.pi * 2 * t)
        signal = np.sin(2 * np.pi * freq * t)
    elif class_idx == 1: # Dog bark: intermittent bursts
        envelope = np.exp(-((t % 0.5 - 0.1) ** 2) / 0.005)
        signal = np.sin(2 * np.pi * 400 * t) * envelope
    elif class_idx == 2: # Drilling: high-freq raspy tone
        signal = np.sin(2 * np.pi * 1200 * t) + rng.normal(0, 0.4, len(t))
    elif class_idx == 3: # Gunshot: sharp transient impulse
        envelope = np.exp(-t / 0.05)
        signal = rng.normal(0, 1, len(t)) * envelope
    else: # Street music: harmonic mixture
        signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)

    noise = rng.normal(0, 0.05, len(t))
    audio = np.clip(signal + noise, -1.0, 1.0).astype(np.float32)
    return audio, sample_rate, class_idx
