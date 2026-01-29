"""
src/models.py
EEG (1D) and Spectrogram (2D) baselines + optional multimodal fusion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class EEG1DCNN(nn.Module):
    """
    Input:  x (B, C, T)  e.g. C=16~20 channels, T=10000 (50s @ 200Hz) or downsampled.
    Output: logits (B, num_classes)
    """

    def __init__(self, in_ch: int = 16, num_classes: int = 6, base: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, base, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base),
            nn.GELU(),
            nn.Conv1d(base, base, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(base, base * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base * 2),
            nn.GELU(),
            nn.Conv1d(base * 2, base * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(base * 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(base * 2, base * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(base * 4),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(base * 4, base * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(base * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class SpectrogramResNet(nn.Module):
    """
    Input: image tensor (B, 1, H, W) or (B, 3, H, W)
    Output: logits (B, num_classes)
    """

    def __init__(self, in_ch: int = 1, num_classes: int = 6):
        super().__init__()
        m = resnet18(weights=None)
        # adapt first conv if needed
        if in_ch != 3:
            m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        self.model = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiModalLateFusion(nn.Module):
    """
    Late fusion:
    - EEG branch -> feature
    - Spectrogram branch -> feature
    then concat and classify
    """

    def __init__(self, eeg_in_ch: int = 16, spec_in_ch: int = 1, num_classes: int = 6, hidden: int = 256):
        super().__init__()
        self.eeg = EEG1DCNN(in_ch=eeg_in_ch, num_classes=hidden)
        self.spec = SpectrogramResNet(in_ch=spec_in_ch, num_classes=hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, eeg: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        fe = self.eeg(eeg)
        fs = self.spec(spec)
        x = torch.cat([fe, fs], dim=1)
        return self.head(x)
