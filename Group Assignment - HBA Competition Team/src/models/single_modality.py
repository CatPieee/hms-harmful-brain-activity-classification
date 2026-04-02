import torch
import torch.nn as nn

from .eeg_1d import EEGResNet1D
from .spec_2d import SpectrogramEfficientNet


class EEGOnlyNet(nn.Module):
    def __init__(self, n_classes: int = 6, in_ch: int = 16, feat_dim: int = 256, drop: float = 0.3):
        super().__init__()
        self.encoder = EEGResNet1D(in_ch=in_ch, feat_dim=feat_dim, drop=drop / 2)
        self.head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(feat_dim, feat_dim),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(feat_dim, n_classes),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(eeg)
        return self.head(feat)


class SpecOnlyNet(nn.Module):
    def __init__(self, n_classes: int = 6, in_ch: int = 1, feat_dim: int = 256, pretrained: bool = True, drop: float = 0.3):
        super().__init__()
        self.encoder = SpectrogramEfficientNet(in_ch=in_ch, feat_dim=feat_dim, pretrained=pretrained, drop=drop / 2)
        self.head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(feat_dim, feat_dim),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(feat_dim, n_classes),
        )

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(spec)
        return self.head(feat)
