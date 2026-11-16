"""Spectral gating noise reduction preprocessor."""
import numpy as np

def reduce_noise_spectral(audio_waveform: np.ndarray, threshold_db: float = -20.0) -> np.ndarray:
    """Zero out spectral frequency bins below ambient noise threshold."""
    fft_vals = np.fft.rfft(audio_waveform)
    magnitudes = np.abs(fft_vals)
    peak = np.max(magnitudes) + 1e-8
    mask = (magnitudes / peak) > 10 ** (threshold_db / 20.0)
    cleaned_fft = fft_vals * mask
    cleaned_waveform = np.fft.irfft(cleaned_fft, n=len(audio_waveform))
    return cleaned_waveform.astype(np.float32)
