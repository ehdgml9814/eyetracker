"""
eval.py — 테스트셋 평가 → result.yaml에 test_angular_err 추가

사용법:
  python src/eval.py --exp-dir /workspace/runs/exp_none_none_none_resnet18
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import save_yaml
from src.utils.metrics import angular_error_np
from src.data.dataset import MPIIGazeDataset


def _build_model(cfg: dict) -> torch.nn.Module:
    crop_selected = cfg.get("category", {}).get("crop", {}).get("selected", "none")
    if crop_selected == "adaptive":
        from src.models.adaptive_filter import GazeEstimatorV2
        return GazeEstimatorV2(cfg)
    else:
        from src.models.gaze_model import GazeEstimator
        return GazeEstimator(cfg)


def evaluate(exp_dir: Path) -> float:
    with open(exp_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    infra       = cfg.get("infrastructure", {})
    num_workers = int(infra.get("num_workers", 4))
    pin_memory  = bool(infra.get("pin_memory", True)) and device.type == "cuda"
    batch_size  = int(cfg.get("train", {}).get("batch_size", 256))

    processed_dir = Path(cfg["dataset"]["processed_dir"])
    test_ds = MPIIGazeDataset(processed_dir / "test.h5", cfg, split="test")
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        **( {"prefetch_factor": 2} if num_workers > 0 else {} ),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs
    )

    model = _build_model(cfg)
    model.load_state_dict(torch.load(exp_dir / "best.pt", map_location=device))
    model = model.to(device)
    model.eval()

    preds_list, gts_list = [], []

    with torch.no_grad():
        for batch in test_loader:
            pred = model(
                batch["left"].to(device),
                batch["right"].to(device),
                batch["head_pose"].to(device),
            ).cpu().numpy()                           # (B,3)
            preds_list.append(pred)
            gts_list.append(batch["gaze"].numpy())    # (B,3)

    preds    = np.concatenate(preds_list)
    gts      = np.concatenate(gts_list)
    test_err = angular_error_np(preds, gts)

    print(f"[eval] test_angular_err = {test_err:.4f}°")

    result_path = exp_dir / "result.yaml"
    result = {}
    if result_path.exists():
        with open(result_path) as f:
            result = yaml.safe_load(f) or {}
    result.setdefault("best", {})["test_angular_err"] = float(test_err)
    save_yaml(result, result_path)

    return test_err


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--config-dir", default=None)  # run.py 호환용 (미사용)
    args = parser.parse_args()
    evaluate(Path(args.exp_dir))


if __name__ == "__main__":
    main()
