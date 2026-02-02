from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal

ModelKind = Literal["yolo", "custom"]

@dataclass(frozen=True)
class ModelSpec:
    kind: ModelKind
    weights: Path

MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "custom_fcos": ModelSpec(kind="custom", weights=Path("models/custom/best.pt")),
    "yolo_fast":   ModelSpec(kind="yolo",   weights=Path("runs/detect/runs_yolo/fast/weights/best.pt")),
    "yolo_new":    ModelSpec(kind="yolo",   weights=Path("runs/detect/runs_yolo/accurate/weights/best.pt")),
}