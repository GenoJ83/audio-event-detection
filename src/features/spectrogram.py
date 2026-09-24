"""Log-Mel Spectrogram and MFCC feature extraction pipeline."""
import numpy as np

def extract_log_mel_spectrogram(waveform: np.ndarray, n_fft: int = 512, hop_length: int = 256, n_mels: int = 64):
    num_frames = (len(waveform) - n_fft) // hop_length + 1
    spec = np.zeros((n_mels, max(1, num_frames)), dtype=np.float32)
    window = np.hanning(n_fft)

    for i in range(num_frames):
        frame = waveform[i * hop_length : i * hop_length + n_fft] * window
        fft = np.abs(np.fft.rfft(frame))[:n_mels]
        spec[:len(fft), i] = np.log(fft + 1e-6)

    return spec
