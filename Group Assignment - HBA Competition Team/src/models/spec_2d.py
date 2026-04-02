import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
except Exception as e:
    efficientnet_b0 = None
    EfficientNet_B0_Weights = None


class SpectrogramEfficientNet(nn.Module):
    """
    EfficientNet-B0 backbone for spectrogram images.
    - Adapts first conv layer to grayscale (1 channel) by averaging RGB weights.
    Input: (B, 1, H, W)
    Output: (B, feat_dim)
    """
    def __init__(self, in_ch: int = 1, feat_dim: int = 256, pretrained: bool = True, drop: float = 0.2):
        super().__init__()
        if efficientnet_b0 is None:
            raise ImportError(
                "torchvision is required for EfficientNet. Install torchvision>=0.16."
            )

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if (pretrained and EfficientNet_B0_Weights) else None
        net = efficientnet_b0(weights=weights)

        # Patch first conv to accept 1-channel input
        if in_ch == 1:
            first = net.features[0][0]  # Conv2d(3, 32, ...)
            w = first.weight.data
            # average over RGB channels -> (32,1,kh,kw)
            w_gray = w.mean(dim=1, keepdim=True)
            new_first = nn.Conv2d(1, first.out_channels, kernel_size=first.kernel_size,
                                  stride=first.stride, padding=first.padding, bias=False)
            new_first.weight.data.copy_(w_gray)
            net.features[0][0] = new_first
        elif in_ch != 3:
            raise ValueError("EfficientNet backbone supports in_ch=1 or 3 in this implementation.")

        self.backbone = net.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(drop)
        self.proj = nn.Linear(net.classifier[1].in_features, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = self.proj(x)
        return x
