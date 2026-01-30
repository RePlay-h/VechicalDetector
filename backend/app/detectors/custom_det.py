from typing import List, Tuple
import numpy as np
import torch
import cv2

from .base import BaseDetector
from ml.models import build_model
from ml.models.postprocess_fcos import decode_fcos
from ml.train import TrainCustomConfig, load_config

class CustomFCOSDetector(BaseDetector):
    def __init__(self, weights_path: str, device: str = "cpu"):
        self.device = torch.device(device)

        p = str(weights_path)
        self.is_jit = p.endswith(".torchscript") or p.endswith(".ts")

        cfg = load_config("params.yaml")
        model = build_model(
                "efficient_pan_af",
                num_classes=cfg.num_classes,
                backbone_name=cfg.backbone,
                pretrained_backbone=cfg.pretrained_backbone,
        ).to(device).eval()

        ckpt = torch.load(p, map_location=self.device)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        self.model.load_state_dict(state, strict=False)
        
        self.decode_fcos = decode_fcos

    def _preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:

        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0
        return t.unsqueeze(0)

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray, score_thr: float, iou_thr: float) -> List[Tuple[float,float,float,float,float,int]]:
        inp = self._preprocess(image_bgr).to(self.device)

        out = self.model(inp)

        dets = self.decode_fcos(
            cls_logits_level=out.cls_logits,
            box_reg_levels=out.box_reg,
            ctr_logits_levels=getattr(out, "centerness", None),
            strides=[8, 16, 32],
            img_size=(image_bgr.shape[0], image_bgr.shape[1]),
            score_thesh=score_thr,
            iou_thresh=iou_thr,
        )[0]

        boxes = dets.boxes.detach().cpu().numpy()
        scores = dets.scores.detach().cpu().numpy()
        labels = dets.labels.detach().cpu().numpy().astype(int)

        out_list: List[Tuple[float,float,float,float,float,int]] = []
        for (x1,y1,x2,y2), s, c in zip(boxes, scores, labels):
            out_list.append((float(x1), float(y1), float(x2), float(y2), float(s), int(c)))
        return out_list