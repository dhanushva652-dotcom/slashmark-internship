from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch

from src.airsim_adapter import AirSimAdapter
from src.control import ObstacleAvoidanceController
from src.model import PolicyNet


def parse_args():
    p = argparse.ArgumentParser(description="Run the obstacle avoidance loop on webcam, video, or AirSim.")
    p.add_argument("--source", type=str, default="0", help="Webcam index or video path.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained policy_net.pt.")
    p.add_argument("--airsim", action="store_true", help="Use AirSim instead of local camera/video.")
    p.add_argument("--display", action="store_true", help="Show annotated frames in a window.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = PolicyNet(max_angle=ckpt.get("max_angle", 0.3141592653589793)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    controller = ObstacleAvoidanceController(model, device, max_angle=model.max_angle)

    airsim_adapter = None
    cap = None

    if args.airsim:
        airsim_adapter = AirSimAdapter()
        if not airsim_adapter.connect():
            raise RuntimeError("AirSim is not installed or could not connect.")
    else:
        source = args.source
        try:
            source = int(source)
        except ValueError:
            pass
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source: {args.source}")

    while True:
        if args.airsim:
            frame = airsim_adapter.get_frame()
            if frame is None:
                continue
        else:
            ok, frame = cap.read()
            if not ok:
                break

        decision = controller.decide(frame)

        if args.airsim and airsim_adapter is not None:
            airsim_adapter.send(decision.speed, decision.angle)
        else:
            # For a local demo, draw the command on the frame.
            label = f"speed={decision.speed:.2f} angle={decision.angle:.3f} risk={decision.obstacle_score:.2f} {decision.reason}"
            cv2.putText(frame, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if args.display:
                cv2.imshow("Indoor Obstacle Avoidance", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if not args.airsim and not args.display:
            print(
                f"speed={decision.speed:.2f} angle={decision.angle:.3f} "
                f"risk={decision.obstacle_score:.2f} reason={decision.reason}"
            )

    if cap is not None:
        cap.release()
    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
