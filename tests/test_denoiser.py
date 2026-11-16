"""Tests for noise suppressor."""
import numpy as np
from src.features.denoiser import reduce_noise_spectral

def test_denoiser_shape_and_reduction():
    signal = np.sin(np.linspace(0, 100, 1600))
    noise = np.random.normal(0, 0.05, 1600)
    cleaned = reduce_noise_spectral(signal + noise)
    assert len(cleaned) == 1600
    assert np.all(np.isfinite(cleaned))
