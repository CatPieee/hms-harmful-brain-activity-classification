import torch
import torch.nn as nn
import torch.nn.functional as F

from .eeg_1d import EEGResNet1D
from .spec_2d import SpectrogramEfficientNet


class GatedFusion(nn.Module):
    """
    Simple gated fusion:
    gate = sigmoid(MLP([eeg, spec]))
    fused = [gate*eeg, (1-gate)*spec] then concat.
    """
    def __init__(self, eeg_dim: int, spec_dim: int, hidden: int = 256, drop: float = 0.2):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(eeg_dim + spec_dim, hidden),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, 1),
        )

    def forward(self, eeg: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([eeg, spec], dim=1)))  # (B,1)
        fused = torch.cat([g * eeg, (1.0 - g) * spec], dim=1)
        return fused


class MultiModalNet(nn.Module):
    """
    Multimodal network for HMS:
    - EEG encoder (1D ResNet)
    - Spectrogram encoder (EfficientNet-B0)
    - Gated fusion + classification head
    """
    def __init__(
        self,
        n_classes: int = 6,
        eeg_in_ch: int = 16,
        eeg_feat: int = 256,
        spec_in_ch: int = 1,
        spec_feat: int = 256,
        pretrained_spec: bool = True,
        drop: float = 0.3,
    ):
        super().__init__()
        self.eeg = EEGResNet1D(in_ch=eeg_in_ch, feat_dim=eeg_feat, drop=drop/2)
        self.spec = SpectrogramEfficientNet(in_ch=spec_in_ch, feat_dim=spec_feat, pretrained=pretrained_spec, drop=drop/2)
        self.fuse = GatedFusion(eeg_feat, spec_feat, hidden=256, drop=drop/2)
        self.head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(eeg_feat + spec_feat, 256),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(256, n_classes),
        )

    def forward(self, eeg: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        eeg_f = self.eeg(eeg)
        spec_f = self.spec(spec)
        fused = self.fuse(eeg_f, spec_f)
        return self.head(fused)
