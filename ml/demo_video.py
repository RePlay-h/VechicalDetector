from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Protocol

import cv2
import torch
import yaml
from tqdm import tqdm
from loguru import logger

from ml.models import build_model
from ml.models.postprocess_fcos import decode_fcos, Detections

from ultralytics import YOLO

import argparse


target_names = {
    0: "car",
    1: "van",
    2: "truck",
    3: "tricycle",
    4: "awning-tricycle",
    5: "bus",
    6: "motor",
}

class Detector(Protocol):
    def predict_frame(self, frame_bgr, imgsz: int) -> Detections:
        ...

def letterbox_bgr(img_bgr, new_size: int = 640, color=(0, 0, 0)):
    h, w = img_bgr.shape[:2]
    scale = min(new_size / w, new_size / h)
    nw = int(round(w * scale))
    nh = int(round(h * scale))

    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_x = new_size - nw
    pad_y = new_size - nh
    pad_left = pad_x // 2
    pad_right = pad_x - pad_left
    pad_top = pad_y // 2
    pad_bottom = pad_y - pad_top

    img_lb = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color)
    return img_lb, scale, pad_left, pad_top


def unletterbox_boxes(boxes_xyxy: torch.Tensor, scale: float, pad_left: int, pad_top: int, w: int, h: int) -> torch.Tensor:
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy
    b = boxes_xyxy.clone()
    b[:, [0, 2]] -= pad_left
    b[:, [1, 3]] -= pad_top
    b /= scale
    # clip to original image
    b[:, 0].clamp_(0, w)
    b[:, 2].clamp_(0, w)
    b[:, 1].clamp_(0, h)
    b[:, 3].clamp_(0, h)
    return b

@dataclass
class CustomCfg:
    weights: Path
    num_classes: int
    backbone: str = "efficientnet_b0"
    pretrained_backbone: bool = False
    strides: Tuple[int, int, int] = (8, 16, 32)
    score_thresh: float = 0.25
    nms_iou: float = 0.5
    device: str = "cpu"

class CustomDetector:
    def __init__(self, cfg: CustomCfg):
        self.cfg = cfg
        device = torch.device(cfg.device)
        if str(cfg.device).startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; using CPU.")
            device = torch.device("cpu")
            self.cfg.device = "cpu"

        self.device = device
        self.model = build_model(
            "efficient_pan_af",
            num_classes=cfg.num_classes,
            backbone_name=cfg.backbone,
            pretrained_backbone=cfg.pretrained_backbone,
        ).to(self.device)

        ckpt = torch.load(cfg.weights, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"], strict=True)
        self.model.eval()

    @torch.no_grad()
    def predict_frame(self, frame_bgr, imgsz: int) -> Detections:
        h0, w0 = frame_bgr.shape[:2]
        lb, scale, pad_left, pad_top = letterbox_bgr(frame_bgr, new_size=imgsz)

        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255.0
        x = x.unsqueeze(0).to(self.device)

        out = self.model(x)
        det = decode_fcos(
            cls_logits_level=out.cls_logits,
            box_reg_levels=out.box_reg,
            ctr_logits_levels=out.centerness,
            strides=list(self.cfg.strides),
            img_size=(imgsz, imgsz),
            score_thesh=self.cfg.score_thresh,
            iou_thresh=self.cfg.nms_iou,
            topk_per_level=2000,
            max_detections=300,
        )[0]

        boxes = det.boxes.detach().cpu()
        scores = det.scores.detach().cpu()
        labels = det.labels.detach().cpu()

        boxes = unletterbox_boxes(boxes, scale, pad_left, pad_top, w=w0, h=h0)
        return Detections(boxes=boxes, scores=scores, labels=labels)
    
@dataclass
class YoloCfg:
    weights: str  # e.g. "yolov8n.pt" or "yolo11n.pt" or path
    conf: float = 0.25
    iou: float = 0.5
    device: str = "cpu"

class YoloDetector:
    def __init__(self, cfg: YoloCfg):
        self.cfg = cfg
        self.model = YOLO(cfg.weights)

    @torch.no_grad()
    def predict_frame(self, frame_bgr, imgsz: int) -> Detections:
        r = self.model.predict(
            source=frame_bgr,
            imgsz=imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            device=self.cfg.device,
            verbose=False
        )[0]

        if r.boxes is None or len(r.boxes) == 0:
            return Detections(
                boxes=torch.zeros((0, 4)),
                scores=torch.zeros((0, )),
                labels=torch.zeros((0, ), dtype=torch.long)
            )
        
        boxes = torch.tensor(r.boxes.xyxy.numpy(), dtype=torch.float32)
        scores = torch.tensor(r.boxes.conf.cpu().numpy(), dtype=torch.float32)
        labels = torch.tensor(r.boxes.cls.cpu().numpy(), dtype=torch.long)
        return Detections(boxes=boxes, scores=scores, labels=labels)
    

def build_detector(params_path: Path, model_name: str) -> Detector:
    raw = yaml.safe_load(params_path.read_text())
    model_name = model_name.lower()

    if model_name == "custom":
        p = raw["demo"]["custom"]
        cfg = CustomCfg(
            weights=Path(p["weights"]),
            num_classes=int(p["num_classes"]),
            backbone=str(p.get("backbone", "efficientnet_b0")),
            pretrained_backbone=bool(p.get("pretrained_backbone", False)),
            strides=tuple(p.get("strides", [8, 16, 32])),
            score_thresh=float(p.get("score_thresh", 0.25)),
            nms_iou=float(p.get("nms_iou", 0.5)),
            device=str(p.get("device", "cpu")),
        )
        return CustomDetector(cfg)

    if model_name.startswith("yolo"):
        p = raw["demo"][model_name]
        cfg = YoloCfg(
            weights=str(p["weights"]),
            conf=float(p.get("conf", 0.25)),
            iou=float(p.get("iou", 0.5)),
            device=str(p.get("device", "cpu")),
        )
        return YoloDetector(cfg)

    raise ValueError(f"Unknown model_name='{model_name}'. Use: custom, yolo_fast, yolo_new")


def draw_detections(frame_bgr, det: Detections, score_thr: float) -> None:
    boxes = det.boxes
    scores = det.scores
    labels = det.labels

    for b, s, c in zip(boxes, scores, labels):
        if float(s) < score_thr:
            continue

        x1, y1, x2, y2 = [int(v) for v in b.tolist()]
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame_bgr,
            f"{c.item()} {float(s):.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

def main() -> None:
    ap = argparse.ArgumentParser()
    
    ap.add_argument("--params", type=Path, default=Path("params.yaml"))
    ap.add_argument("--model", type=str, required=True, help="custom | yolo_fast | yolo_new")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("runs/demo_out.mp4"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--draw-thr", type=float, default=0.25)
    ap.add_argument("--max-frames", type=int, default=-1)

    args = ap.parse_args()

    detector = build_detector(args.params, args.model)

    cap = cv2.VideoCapture(str(args.input))
    
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.input}")
    

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    outv = cv2.VideoWriter(str(args.output), fourcc, fps, (w, h))

    logger.info(f"model={args.model} imgsz={args.imgsz} draw_thr={args.draw_thr}")
    logger.info(f"input={args.input} ({w}x{h} fps={fps:.2f} frames={n_frames})")
    logger.info(f"output={args.output}")

    frame_idx = 0
    pbar = tqdm(total=(n_frames if n_frames > 0 else None), desc="processing", leave=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        det = detector.predict_frame(frame, imgsz=args.imgsz)
        draw_detections(frame, det, score_thr=args.draw_thr)
        outv.write(frame)

        frame_idx += 1
        pbar.update(1)
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

    pbar.close()
    cap.release()
    outv.release()
    logger.info("Done.")


if __name__ == "__main__":
    main()
