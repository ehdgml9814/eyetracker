"""
train.py — 단일 실험 학습

손실 함수: 1 - cosine_similarity(pred, gt)
           = 1 - dot(pred_unit, gt_unit)
           ≈ angular error의 단조 함수 → angular error 직접 최소화

사용법:
  python src/train.py \\
      --config-dir /workspace/configs \\
      --exp-dir /workspace/runs/exp_none_none_none_resnet18 \\
      --set category.det.selected=clahe \\
      --set model.backbone=resnet18
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config, save_yaml, append_yaml_log
from src.utils.metrics import angular_error_batch
from src.data.dataset import MPIIGazeDataset


# def _build_model(cfg: dict) -> nn.Module:
#     crop_selected = cfg.get("category", {}).get("crop", {}).get("selected", "none")
#     if crop_selected == "adaptive":
#         from src.models.adaptive_filter import GazeEstimatorV2
#         return GazeEstimatorV2(cfg)
#     else:
#         from src.models.gaze_model import GazeEstimator
#         return GazeEstimator(cfg)
def _build_model(cfg: dict) -> nn.Module:
    model_type = cfg.get("model", {}).get("type", "proposed")
    if model_type == "proposed":
        from src.models.proposed_model import ProposedModel
        return ProposedModel(cfg)
    elif model_type == "baseline":
        from src.models.baseline_model import BaselineModel
        return BaselineModel(cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from [proposed, baseline]")


def cosine_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    1 - cosine_similarity 평균.
    pred: (B,3) 이미 normalize된 단위벡터
    gt  : (B,3) 단위벡터
    값 범위: [0, 2] — 0이면 완벽하게 일치
    """
    return (1.0 - F.cosine_similarity(pred, gt, dim=1)).mean()


def _build_scheduler(optimizer, cfg: dict):
    train_cfg      = cfg.get("train", {})
    scheduler_name = train_cfg.get("lr_scheduler", "cosine")
    epochs         = int(train_cfg.get("epochs", 50))
    warmup         = int(train_cfg.get("lr_warmup_epochs", 0))

    if scheduler_name == "cosine":
        main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup, 1)
        )
    elif scheduler_name == "step":
        main_sched = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(epochs // 3, 1), gamma=0.1
        )
    elif scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        ), "plateau"
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    if warmup > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup_sched, main_sched], milestones=[warmup]
        )
        return sched, "sequential"

    return main_sched, scheduler_name


def train(cfg: dict, exp_dir: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    infra       = cfg.get("infrastructure", {})
    num_workers = int(infra.get("num_workers", 4))
    pin_memory  = bool(infra.get("pin_memory", True)) and device.type == "cuda"

    train_cfg  = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 256))
    epochs     = int(train_cfg.get("epochs", 50))
    lr         = float(train_cfg.get("lr", 1e-4))
    wd         = float(train_cfg.get("weight_decay", 1e-4))

    processed_dir = Path(cfg["dataset"]["processed_dir"])

    train_ds = MPIIGazeDataset(processed_dir / "train.h5", cfg, split="train")
    val_ds   = MPIIGazeDataset(processed_dir / "val.h5",   cfg, split="val")

    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        **( {"prefetch_factor": 2} if num_workers > 0 else {} ),
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs
    )

    model     = _build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler, sched_type = _build_scheduler(optimizer, cfg)

    train_mode   = cfg.get("train", {}).get("mode", "e2e")
    phase1_ratio = float(cfg.get("train", {}).get("phase1_ratio", 0.4))
    phase1_end   = int(epochs * phase1_ratio)

    best_val_err = float("inf")
    best_epoch   = 0
    log_path     = exp_dir / "train_log.yaml"

    for epoch in range(1, epochs + 1):
        # ── Sequential Phase2 진입 처리 ──────────────────────────────────────
        if train_mode == "sequential" and epoch == phase1_end + 1:
            if hasattr(model, "kernel_net"):
                for p in model.kernel_net.parameters():
                    p.requires_grad = False
                # optimizer 재빌드 (frozen params 제외)
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=lr, weight_decay=wd,
                )
                scheduler, sched_type = _build_scheduler(optimizer, cfg)
                print(f"[train] Sequential Phase2 시작 (epoch {epoch}): KernelNet freeze")

        # ── 학습 ──────────────────────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0

        for batch in train_loader:
            left      = batch["left"].to(device)
            right     = batch["right"].to(device)
            head_pose = batch["head_pose"].to(device)
            gaze      = batch["gaze"].to(device)      # (B,3) 단위벡터

            optimizer.zero_grad()
            pred = model(left, right, head_pose)      # (B,3) 단위벡터
            loss = cosine_loss(pred, gaze)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_sum += loss.item() * left.size(0)

        train_loss = train_loss_sum / len(train_ds)

        # ── 검증 ──────────────────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_preds, val_gts = [], []

        with torch.no_grad():
            for batch in val_loader:
                left      = batch["left"].to(device)
                right     = batch["right"].to(device)
                head_pose = batch["head_pose"].to(device)
                gaze      = batch["gaze"].to(device)

                pred = model(left, right, head_pose)
                val_loss_sum += cosine_loss(pred, gaze).item() * left.size(0)
                val_preds.append(pred.cpu())
                val_gts.append(gaze.cpu())

        val_loss    = val_loss_sum / len(val_ds)
        val_pred_all = torch.cat(val_preds)   # (N,3)
        val_gt_all   = torch.cat(val_gts)     # (N,3)
        val_err      = angular_error_batch(val_pred_all, val_gt_all)

        if sched_type == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_err={val_err:.2f}° lr={current_lr:.2e}"
        )

        append_yaml_log(
            {
                "epoch":           epoch,
                "lr":              float(current_lr),
                "train_loss":      float(train_loss),
                "val_loss":        float(val_loss),
                "val_angular_err": float(val_err),
            },
            log_path,
        )

        if val_err < best_val_err:
            best_val_err = val_err
            best_epoch   = epoch
            torch.save(model.state_dict(), exp_dir / "best.pt")

    print(f"[train] 최적 epoch={best_epoch}  val_err={best_val_err:.2f}°")

    # save_yaml(
    #     {
    #         "experiment": {
    #             "det_filter":  cfg.get("category", {}).get("det",  {}).get("selected", "none"),
    #             "pose_filter": cfg.get("category", {}).get("pose", {}).get("selected", "none"),
    #             "crop_filter": cfg.get("category", {}).get("crop", {}).get("selected", "none"),
    #             "backbone":    cfg.get("model", {}).get("backbone", "resnet18"),
    #         },
    #         "best": {
    #             "epoch":           best_epoch,
    #             "val_angular_err": float(best_val_err),
    #         },
    #     },
    #     exp_dir / "result.yaml",
    # )
    save_yaml(
        {
            "experiment": {
                "model_type":       cfg.get("model", {}).get("type", "proposed"),
                "train_mode":       cfg.get("train", {}).get("mode", "e2e"),
                "kernel_hidden":    cfg.get("model", {}).get("kernel_hidden", 0),
                "regressor_hidden": cfg.get("model", {}).get("regressor_hidden", 256),
            },
            "best": {
                "epoch":           best_epoch,
                "val_angular_err": float(best_val_err),
            },
        },
        exp_dir / "result.yaml",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="/workspace/configs")
    parser.add_argument("--exp-dir",    required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg     = load_config(args.config_dir, args.overrides)
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, exp_dir / "config.yaml")

    t0 = time.time()
    train(cfg, exp_dir)
    print(f"[train] 완료 ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
