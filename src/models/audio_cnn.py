"""2D Spectrogram Convolutional Neural Network."""
import torch
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x):
        feat = self.conv(x)
        logits = self.fc(torch.flatten(feat, 1))
        return logits
