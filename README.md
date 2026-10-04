# Audio Event Detection & Environmental Sound Classification

![PyTorch](https://img.shields.io/badge/PyTorch-Audio-red)
![DSP](https://img.shields.io/badge/DSP-Log--Mel--Spectrograms-blue)
![Augment](https://img.shields.io/badge/Augment-SpecAugment-green)
![License](https://img.shields.io/badge/License-MIT-green)

An acoustic pattern recognition system recognizing environmental acoustic events (sirens, barking, machinery, gunshots, music) from audio streams.

## Audio Processing Pipeline
```mermaid
flowchart LR
    Wave[Raw Audio PCM] --> STFT[Short-Time Fourier Transform]
    STFT --> Mel[Log-Mel Spectrogram]
    Mel --> CNN[2D ResNet / AudioCNN]
    CNN --> Out[Class Event Probabilities]
```

## Quick Test
```bash
python src/cli/audio_detect.py --class-sim 1
```
