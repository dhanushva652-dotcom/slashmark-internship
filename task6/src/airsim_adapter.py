from __future__ import annotations

from dataclasses import dataclass

try:
    import airsim  # type: ignore
except Exception:  # pragma: no cover
    airsim = None


@dataclass
class AirSimCommand:
    speed: float
    angle: float
    reason: str


class AirSimAdapter:
    def __init__(self):
        self.available = airsim is not None
        self.client = None

    def connect(self):
        if not self.available:
            return False

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        if self.client.getMultirotorState().landed_state == airsim.LandedState.Landed:
            self.client.takeoffAsync().join()
        return True

    def get_frame(self):
        if not self.available or self.client is None:
            return None
        raw = self.client.simGetImage("0", airsim.ImageType.Scene)
        if raw is None:
            return None
        import cv2
        import numpy as np

        img = cv2.imdecode(airsim.string_to_uint8_array(raw), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return img

    def send(self, speed: float, angle: float, duration: float = 0.4):
        if not self.available or self.client is None:
            return
        import math
        import airsim

        pitch, roll, yaw = airsim.to_eularian_angles(self.client.simGetVehiclePose().orientation)
        yaw = yaw + angle
        vx = speed * math.cos(yaw)
        vy = speed * math.sin(yaw)
        self.client.moveByVelocityZAsync(
            vx, vy, -2, duration, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0)
        ).join()
