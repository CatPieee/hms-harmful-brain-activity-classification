import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, drop: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

        self.proj = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        x = F.silu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.bn2(self.conv2(x))
        x = F.silu(x + identity)
        return x


class EEGResNet1D(nn.Module):
    """
    Competition-style 1D encoder for raw EEG.
    Input: (B, C, L)
    Output: (B, feat_dim)
    """
    def __init__(self, in_ch: int = 16, feat_dim: int = 256, base: int = 64, drop: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(base),
            nn.SiLU(),
        )
        self.layer1 = nn.Sequential(
            ResidualBlock1D(base, base, stride=1, drop=drop),
            ResidualBlock1D(base, base, stride=1, drop=drop),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock1D(base, base * 2, stride=2, drop=drop),
            ResidualBlock1D(base * 2, base * 2, stride=1, drop=drop),
        )
        self.layer3 = nn.Sequential(
            ResidualBlock1D(base * 2, base * 4, stride=2, drop=drop),
            ResidualBlock1D(base * 4, base * 4, stride=1, drop=drop),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(base * 4, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        x = self.head(x)
        return x
