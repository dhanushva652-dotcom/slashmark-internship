from __future__ import annotations

import torch
from torch import nn


class PolicyNet(nn.Module):
    """Predicts normalized speed and steering angle from an RGB frame.

    Output format:
        y[:, 0] -> speed in [0, 1]
        y[:, 1] -> angle in [-max_angle, max_angle]
    """

    def __init__(self, max_angle: float = 0.3141592653589793):
        super().__init__()
        self.max_angle = float(max_angle)

        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(36),
            nn.ReLU(inplace=True),

            nn.Conv2d(36, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),

            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        raw = self.head(z)
        speed = torch.sigmoid(raw[:, :1])
        angle = torch.tanh(raw[:, 1:2]) * self.max_angle
        return torch.cat([speed, angle], dim=1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[float, float]:
        self.eval()
        y = self(x.unsqueeze(0))
        return float(y[0, 0].item()), float(y[0, 1].item())
