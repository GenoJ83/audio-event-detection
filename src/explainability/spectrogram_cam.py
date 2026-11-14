"""Class activation attribution for salient frequency bands."""
import numpy as np

def extract_salient_frequencies(spectrogram: np.ndarray):
    band_energies = np.mean(spectrogram, axis=1)
    dominant_band = int(np.argmax(band_energies))
    return {"dominant_mel_band": dominant_band, "band_energy_profile": band_energies.tolist()}
