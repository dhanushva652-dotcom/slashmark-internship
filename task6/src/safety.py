from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SafetyMetrics:
    obstacle_score: float
    left_clutter: float
    center_clutter: float
    right_clutter: float
    turn_bias: float


def _edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 70, 160)
    return float(edges.mean() / 255.0)


def analyze_frame(frame_bgr: np.ndarray) -> SafetyMetrics:
    """Estimate obstacle risk using simple but effective image clutter heuristics.

    The score is not a semantic detector. It is a safe reactive layer that
    complements the learned policy.
    """

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    h, w = gray.shape
    roi = gray[int(h * 0.45):, :]
    if roi.size == 0:
        return SafetyMetrics(1.0, 1.0, 1.0, 1.0, 0.0)

    third = max(1, roi.shape[1] // 3)
    left = roi[:, :third]
    center = roi[:, third:2 * third]
    right = roi[:, 2 * third:]

    left_clutter = _edge_density(left)
    center_clutter = _edge_density(center)
    right_clutter = _edge_density(right)

    # Normalize to a practical [0, 1] range.
    avg = 0.20 * left_clutter + 0.50 * center_clutter + 0.20 * right_clutter
    obstacle_score = float(np.clip((avg - 0.015) / 0.08, 0.0, 1.0))

    # Positive turn_bias means steer right, negative means steer left.
    turn_bias = float(np.clip(right_clutter - left_clutter, -1.0, 1.0))
    return SafetyMetrics(obstacle_score, left_clutter, center_clutter, right_clutter, turn_bias)


def safe_action(
    predicted_speed: float,
    predicted_angle: float,
    metrics: SafetyMetrics,
    max_angle: float = 0.3141592653589793,
) -> tuple[float, float, str]:
    """Fuse learned policy with reactive safety heuristics."""

    speed = float(np.clip(predicted_speed, 0.0, 1.0))
    angle = float(np.clip(predicted_angle, -max_angle, max_angle))
    reason = "policy"

    if metrics.obstacle_score < 0.45:
        return speed, angle, reason

    # Slow down as clutter rises.
    speed = speed * (1.0 - 0.65 * metrics.obstacle_score)

    # Bias away from the more cluttered side.
    angle = angle + (-0.50 * metrics.turn_bias * max_angle)

    if metrics.obstacle_score >= 0.70:
        speed = min(speed, 0.25)
        angle = float(np.clip(angle, -max_angle, max_angle))
        reason = "caution"
    if metrics.obstacle_score >= 0.88:
        speed = 0.0
        angle = float(np.clip(0.75 * np.sign(-metrics.turn_bias) * max_angle, -max_angle, max_angle))
        reason = "emergency_stop"

    return float(np.clip(speed, 0.0, 1.0)), float(np.clip(angle, -max_angle, max_angle)), reason
