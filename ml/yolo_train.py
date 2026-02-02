from dataclasses import dataclass
from pathlib import Path
import yaml
from loguru import logger

from ultralytics import YOLO 

from typing import List

import argparse

@dataclass
class YoloTrainCfg:
    data_root: Path
    nc: int
    names: List[str]

    model: str
    imgsz: int
    epochs: int
    batch: int
    device: str
    workers: int
    project: Path
    name: str

    lr0: float = None
    mosaic: float = None
    mixup: float = None

def load_cfg(param_path: Path, key: str) -> YoloTrainCfg:
    raw = yaml.safe_load(param_path.read_text())
    p = raw[key]

    names = p.get("names", None)
    print(p)
    nc = int(p["nc"]) if "nc" in p else (len(names) if names else None)

    return YoloTrainCfg(
        data_root=Path(p["data_root"]),
        nc=int(nc),
        names=names,
        model=str(p["model"]),
        imgsz=int(p.get("imgsz", 640)),
        epochs=int(p.get("epochs", 50)),
        batch=int(p.get("batch", 16)),
        device=str(p.get("device", "cpu")),
        workers=int(p.get("workers", 4)),
        project=str(p.get("project", "runs_yolo")),
        name=str(p.get("name", key)),
        lr0=p.get("lr0", None),
        mosaic=p.get("mosaic", None),
        mixup=p.get("mixup", None),
    )

def main() -> None:
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=Path, default=Path("params.yaml"))
    ap.add_argument("--cfg-key", type=str, required=True, help="yolo_fast or yolo_new")
    args = ap.parse_args()

    cfg = load_cfg(args.params, args.cfg_key)

    data_yaml = cfg.data_root

    logger.info(f"data.yaml written: {data_yaml}")

    model = YOLO(cfg.model)

    print(cfg.project)
    train_kwargs = dict(
        data=str(data_yaml),
        imgsz=cfg.imgsz,
        epochs=cfg.epochs,
        batch=cfg.batch,
        device=cfg.device,
        workers=cfg.workers,
        project=cfg.project,
        name=cfg.name,
        exist_ok=True
    )

    if cfg.lr0 is not None:
        train_kwargs["lr0"] = float(cfg.lr0)
    if cfg.mosaic is not None:
        train_kwargs["mosaic"] = float(cfg.mosaic)
    if cfg.mixup is not None:
        train_kwargs["mixup"] = float(cfg.mixup)

    logger.info(f"Training YOLO: {cfg.model} -> {cfg.project}/{cfg.name}")

    results = model.train(**train_kwargs)



if __name__ == "__main__":
    main()

