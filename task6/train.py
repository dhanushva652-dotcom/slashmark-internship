from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import IndoorObstacleDataset, find_dataset_paths
from src.model import PolicyNet


def parse_args():
    p = argparse.ArgumentParser(description="Train an indoor obstacle avoidance policy network.")
    p.add_argument("--dataset-root", type=str, required=True, help="Path to the extracted reference project.")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def split_indices(n: int, val_size: float = 0.15, seed: int = 42):
    idx = list(range(n))
    train_idx, val_idx = train_test_split(idx, test_size=val_size, random_state=seed, shuffle=True)
    return train_idx, val_idx


def subset_dataset(dataset, indices):
    from torch.utils.data import Subset
    return Subset(dataset, indices)


def main():
    args = parse_args()
    paths = find_dataset_paths(args.dataset_root)

    full_ds = IndoorObstacleDataset(paths.csv_path, paths.images_zip_path, train=True)
    train_idx, val_idx = split_indices(len(full_ds))

    train_ds = subset_dataset(full_ds, train_idx)
    val_ds = subset_dataset(IndoorObstacleDataset(paths.csv_path, paths.images_zip_path, train=False), val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device)
    model = PolicyNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.SmoothL1Loss()

    best_val = float("inf")
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "max_angle": 0.3141592653589793,
        "image_size": [144, 256],
        "targets": ["speed", "angle"],
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)

            # Normalize targets to the same range as the network output.
            y_norm = torch.stack([y[:, 0].clamp(0.0, 1.0), y[:, 1]], dim=1)
            y_norm[:, 1] = y_norm[:, 1].clamp(-model.max_angle, model.max_angle)

            loss = criterion(pred, y_norm)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                y_norm = torch.stack([y[:, 0].clamp(0.0, 1.0), y[:, 1]], dim=1)
                y_norm[:, 1] = y_norm[:, 1].clamp(-model.max_angle, model.max_angle)
                loss = criterion(pred, y_norm)
                val_loss += loss.item() * x.size(0)

        train_loss /= len(train_ds)
        val_loss /= len(val_ds)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_class": "PolicyNet",
                    "max_angle": model.max_angle,
                    "val_loss": best_val,
                },
                ckpt_dir / "policy_net.pt",
            )
            with open(ckpt_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

    print(f"Best checkpoint saved to {ckpt_dir / 'policy_net.pt'}")


if __name__ == "__main__":
    main()
