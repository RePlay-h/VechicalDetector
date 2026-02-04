import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
import yaml
import mlflow

from ml.models import build_model
from ml.models.assigner_fcos import build_fcos_targets
from ml.models.losses_fcos import compute_fcos_losses
from ml.data.yolo_dataset import YoloDetectionDataset
from ml.data.collate import detection_collate

from torchmetrics.detection import MeanAveragePrecision

from ml.models.postprocess_fcos import decode_fcos

import albumentations as A

from dotenv import load_dotenv
import os

load_dotenv()

@dataclass
class TrainCustomConfig:
    data_root: Path
    num_classes: int
    output_dir: Path

    backbone: str = "efficientnet_b0"
    pretrained_backbone: bool = True

    train_split: str = "train"
    val_split: str = "val"

    batch_size: int = 4
    num_workers: int = 0
    epochs: int = 10

    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    strides: Tuple[int, int, int] = (8, 16, 32)
    center_sampling_radius: float = 1.5
    object_sizes_of_interest: Tuple[Tuple[float, float], ...] = ((0.0, 64.0), (64.0, 128.0), (128.0, 1e8))

    # Mixed precision
    amp: bool = True

    imgsz: int = 640
    score_thr: float = 0.25
    nms_iou: float = 0.5

    # MLflow
    mlflow_uri: str = ""
    experiment_name: str = "custom_effnet_pan_af"
    run_name: str = "train_custom"

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _to_device_list(xs: List[torch.Tensor], device: torch.device) -> List[torch.Tensor]:
    return [x.to(device) for x in xs]

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: TrainCustomConfig,
    scheduler: torch.optim.lr_scheduler.StepLR | None,
    scaler: torch.amp.GradScaler,
    epoch: int,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_cls = 0.0
    total_reg = 0.0
    total_ctr = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"train epoch {epoch}", leave=False)

    for images, gt_boxes, gt_labels, _, _ in pbar:
        images = images.to(device)
        gt_boxes = _to_device_list(gt_boxes, device)
        gt_labels = _to_device_list(gt_labels, device)

        out = model(images)
        feats_shapes = [(t.shape[2], t.shape[3]) for t in out.cls_logits]

        targets = build_fcos_targets(
            feats_shapes=feats_shapes,
            strides=list(cfg.strides),
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            num_classes=cfg.num_classes,
            center_sampling_radius=cfg.center_sampling_radius,
            object_sizes_of_interest=list(cfg.object_sizes_of_interest),
        )

        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            losses = compute_fcos_losses(out, targets, strides=list(cfg.strides))

        scaler.scale(losses.total).backward() 
        scaler.step(optimizer)
        scaler.update()
        

        total_loss += float(losses.total.item())
        total_cls += float(losses.cls.item())
        total_reg += float(losses.reg.item())
        total_ctr += float(losses.ctr.item())
        n_batches += 1

        pbar.set_postfix(
            loss=f"{losses.total.item():.4f}",
            cls=f"{losses.cls.item():.4f}",
            reg=f"{losses.reg.item():.4f}",
            ctr=f"{losses.ctr.item():.4f}",
        )

    scheduler.step()

    denom = max(n_batches, 1)
    return {
        "loss_total": total_loss / denom,
        "loss_cls": total_cls / denom,
        "loss_reg": total_reg / denom,
        "loss_ctr": total_ctr / denom,
    }


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainCustomConfig,
    epoch: int,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_cls = 0.0
    total_reg = 0.0
    total_ctr = 0.0
    n_batches = 0

    map_metrics = MeanAveragePrecision(box_format="xyxy", iou_type="bbox").to(device)
    map_metrics.reset()

    pbar = tqdm(loader, desc=f"val epoch {epoch}", leave=False)
    for images, gt_boxes, gt_labels, _, _ in pbar:
        images = images.to(device)
        gt_boxes = _to_device_list(gt_boxes, device)
        gt_labels = _to_device_list(gt_labels, device)

        out = model(images)
        feats_shapes = [(t.shape[2], t.shape[3]) for t in out.cls_logits]

        targets = build_fcos_targets(
            feats_shapes=feats_shapes,
            strides=list(cfg.strides),
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            num_classes=cfg.num_classes,
            center_sampling_radius=cfg.center_sampling_radius,
            object_sizes_of_interest=list(cfg.object_sizes_of_interest),
        )

        dets = decode_fcos(
            cls_logits_level=out.cls_logits,
            box_reg_levels=out.box_reg,
            ctr_logits_levels=out.centerness,
            strides=list(cfg.strides),
            img_size=(cfg.imgsz, cfg.imgsz),
            score_thesh=cfg.score_thr, 
            iou_thresh=cfg.nms_iou,
            topk_per_level=2000,
            max_detections=300,
        )

        preds = []
        tgts = []

        B = len(dets)

        

        for i in range(B):
            preds.append({
                "boxes": dets[i].boxes.to(device).float().reshape(-1, 4),
                "scores": dets[i].scores.to(device).float().reshape(-1),
                "labels": dets[i].labels.to(device).long().reshape(-1)
            })

            tgts.append({
                "boxes": gt_boxes[i].to(device).float().reshape(-1, 4),
                "labels": gt_labels[i].to(device).long().reshape(-1)
            })

        map_metrics.update(preds, tgts)

        losses = compute_fcos_losses(out, targets, strides=list(cfg.strides))

        total_loss += float(losses.total.item())
        total_cls += float(losses.cls.item())
        total_reg += float(losses.reg.item())
        total_ctr += float(losses.ctr.item())
        n_batches += 1

        pbar.set_postfix(loss=f"{losses.total.item():.4f}")

    map_out = map_metrics.compute()
    map_metrics.reset()

    denom = max(n_batches, 1)
    return {
        "loss_total": total_loss / denom,
        "loss_cls": total_cls / denom,
        "loss_reg": total_reg / denom,
        "loss_ctr": total_ctr / denom,
        "map_50": float(map_out["map_50"].item())
    }

def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_val: float) -> None:
    _ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val": best_val,
        },
        path,
    )

def get_albumentations(is_train: bool = True) -> A.Compose:

    if is_train:
        return A.Compose([
            A.GaussianBlur(blur_limit=5, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),)
        ])

    else:
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),)
        ])

def load_config(params_path: Path) -> TrainCustomConfig:
    data = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    cfg_raw: Dict[str, Any] = data["train_custom"]

    return TrainCustomConfig(
        data_root=Path(cfg_raw["data_root"]),
        num_classes=int(cfg_raw["num_classes"]),
        output_dir=Path(cfg_raw["output_dir"]),
        backbone=str(cfg_raw.get("backbone", "efficientnet_b3")),
        pretrained_backbone=bool(cfg_raw.get("pretrained_backbone", True)),
        train_split=str(cfg_raw.get("train_split", "train")),
        val_split=str(cfg_raw.get("val_split", "val")),
        batch_size=int(cfg_raw.get("batch_size", 4)),
        num_workers=int(cfg_raw.get("num_workers", 0)),
        epochs=int(cfg_raw.get("epochs", 10)),
        lr=float(cfg_raw.get("lr", 1e-4)),
        weight_decay=float(cfg_raw.get("weight_decay", 1e-4)),
        grad_clip=float(cfg_raw.get("grad_clip", 5.0)),
        device=str(cfg_raw.get("device", "cuda" if torch.cuda.is_available() else "cpu")),
        seed=int(cfg_raw.get("seed", 42)),
        strides=tuple(cfg_raw.get("strides", [8, 16, 32])),
        center_sampling_radius=float(cfg_raw.get("center_sampling_radius", 1.5)),
        object_sizes_of_interest=tuple(tuple(x) for x in cfg_raw.get("object_sizes_of_interest", [(0.0,64.0),(64.0,128.0),(128.0,1e8)])),
        amp=bool(cfg_raw.get("amp", True)),
        mlflow_uri=str(cfg_raw.get("mlflow_uri")),
        experiment_name=str(cfg_raw.get("experiment_name", "custom_effnet_pan_af")),
        run_name=str(cfg_raw.get("run_name", "train_custom")),
    )

def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.params)

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    _ensure_dir(cfg.output_dir)
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    logger.info(f"Config: {cfg}\n")
    logger.info(f"Device: {device}\n")

    # Datasets
    train_ds = YoloDetectionDataset(root=cfg.data_root, split=cfg.train_split, transfroms=get_albumentations())
    val_ds = YoloDetectionDataset(root=cfg.data_root, split=cfg.val_split, transfroms=get_albumentations(is_train=False))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=detection_collate,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=detection_collate,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = build_model(
        "efficient_pan_af",
        num_classes=cfg.num_classes,
        backbone_name=cfg.backbone,
        pretrained_backbone=cfg.pretrained_backbone,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
    scaler = torch.amp.GradScaler("cuda", (device.type=="cuda"))

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    best_val = math.inf
    best_path = cfg.output_dir / "best.pt"
    last_path = cfg.output_dir / "last.pt"

    with mlflow.start_run(run_name=cfg.run_name):

        mlflow.log_params(
            {
                "backbone": cfg.backbone,
                "pretrained_backbone": cfg.pretrained_backbone,
                "num_classes": cfg.num_classes,
                "batch_size": cfg.batch_size,
                "epochs": cfg.epochs,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "center_sampling_radius": cfg.center_sampling_radius,
                "strides": list(cfg.strides),
                "object_sizes_of_interest": [list(x) for x in cfg.object_sizes_of_interest],
                "amp": cfg.amp,
                "seed": cfg.seed,
            }
        )

        for epoch in range(1, cfg.epochs + 1):
            logger.info(f"\nEpoch {epoch}/{cfg.epochs}\n")

            train_m = train_one_epoch(model, train_loader, optimizer, device, cfg, scheduler, scaler, epoch)
            val_m = validate_one_epoch(model, val_loader, device, cfg, epoch)

            # Log to MLflow
            mlflow.log_metrics({f"train/{k}": v for k, v in train_m.items()}, step=epoch)
            mlflow.log_metrics({f"val/{k}": v for k, v in val_m.items()}, step=epoch)
            mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch)

            logger.info(
                f"train: total={train_m['loss_total']:.4f} cls={train_m['loss_cls']:.4f} "
                f"reg={train_m['loss_reg']:.4f} ctr={train_m['loss_ctr']:.4f}\n"
                
            )
            logger.info(
                f"val:   total={val_m['loss_total']:.4f} cls={val_m['loss_cls']:.4f} "
                f"reg={val_m['loss_reg']:.4f} ctr={val_m['loss_ctr']:.4f}\n"
                f"map_50={val_m["map_50"]}"
            )

            # Save last
            save_checkpoint(last_path, model, optimizer, epoch, best_val)

            if val_m["loss_total"] < best_val:
                best_val = val_m["loss_total"]
                save_checkpoint(best_path, model, optimizer, epoch, best_val)
                logger.info(f"New best: val_loss={best_val:.4f} -> saved {best_path}\n")

        # Log artifacts (checkpoints)
        mlflow.log_artifact(str(best_path))
        mlflow.log_artifact(str(last_path))

    logger.info("Training finished.\n")

if __name__ == "__main__":
    main()