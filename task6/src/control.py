from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .safety import analyze_frame, safe_action


@dataclass
class ControlDecision:
    speed: float
    angle: float
    obstacle_score: float
    reason: str


class ObstacleAvoidanceController:
    def __init__(self, model, device: torch.device, max_angle: float = 0.3141592653589793):
        self.model = model
        self.device = device
        self.max_angle = max_angle

    def preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (256, 144), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(frame).float() / 255.0
        tensor = tensor.permute(2, 0, 1).contiguous()
        return tensor

    @torch.no_grad()
    def decide(self, frame_bgr: np.ndarray) -> ControlDecision:
        self.model.eval()
        metrics = analyze_frame(frame_bgr)

        x = self.preprocess(frame_bgr).unsqueeze(0).to(self.device)
        y = self.model(x)[0].detach().cpu().numpy()
        predicted_speed = float(y[0])
        predicted_angle = float(y[1])

        speed, angle, reason = safe_action(
            predicted_speed=predicted_speed,
            predicted_angle=predicted_angle,
            metrics=metrics,
            max_angle=self.max_angle,
        )
        return ControlDecision(
            speed=speed,
            angle=angle,
            obstacle_score=metrics.obstacle_score,
            reason=reason,
        )
