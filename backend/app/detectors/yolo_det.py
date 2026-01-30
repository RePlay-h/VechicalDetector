from typing import List, Tuple
import numpy as np
from ultralytics import YOLO
from .base import BaseDetector

class YoloDetector(BaseDetector):
    def __init__(self, weights_path: str):
        self.model = YOLO(weights_path)

    def predict(self, image_bgr: np.ndarray, score_thr: float, iou_thr: float) -> List[Tuple[float,float,float,float,float,int]]:
        
        res = self.model.predict(
            source=image_bgr,
            conf=score_thr,
            iou=iou_thr,
            verbose=False,
        )[0]

        out: List[Tuple[float,float,float,float,float,int]] = []
        if res.boxes is None:
            return out

        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        labels = res.boxes.cls.cpu().numpy().astype(int)

        for (x1,y1,x2,y2), s, c in zip(boxes, scores, labels):
            out.append((float(x1), float(y1), float(x2), float(y2), float(s), int(c)))
        return out