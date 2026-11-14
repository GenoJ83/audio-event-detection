"""SpecAugment: Time and frequency spectrogram masking."""
import numpy as np

def apply_spec_augment(spectrogram: np.ndarray, freq_mask_param: int = 8, time_mask_param: int = 16):
    spec = spectrogram.copy()
    num_mels, num_steps = spec.shape

    # Frequency masking
    f = np.random.randint(0, freq_mask_param)
    f0 = np.random.randint(0, max(1, num_mels - f))
    spec[f0:f0 + f, :] = 0

    # Time masking
    t = np.random.randint(0, time_mask_param)
    t0 = np.random.randint(0, max(1, num_steps - t))
    spec[:, t0:t0 + t] = 0

    return spec
