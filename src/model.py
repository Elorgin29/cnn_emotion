"""
CNN léger pour FER2013 (48x48, 1 canal, 7 classes).
683 751 paramètres entraînables — conçu pour tourner sur CPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.constants import IMAGE_SIZE, NUM_CLASSES


class EmotionCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dropout_conv: float = 0.25,
                 dropout_fc: float = 0.5):
        super().__init__()

        def block(in_ch: int, out_ch: int, drop: float) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout(drop),
            )

        self.features = nn.Sequential(
            block(1, 32, dropout_conv),     # 48 -> 24
            block(32, 64, dropout_conv),    # 24 -> 12
            block(64, 128, dropout_conv),   # 12 -> 6
        )
        feat_size = IMAGE_SIZE // 8         # 6
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * feat_size * feat_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = EmotionCNN()
    print(f"Paramètres : {count_parameters(model):,}")
    x = torch.randn(4, 1, IMAGE_SIZE, IMAGE_SIZE)
    print(f"Output shape : {tuple(model(x).shape)}")
