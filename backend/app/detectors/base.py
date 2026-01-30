from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

class BaseDetector(ABC):
    @abstractmethod
    def predict(self, image_bgr: np.ndarray, score_thr: float, iou_thr: float) -> List[Tuple[float,float,float,float,float,int]]:
        raise NotImplementedError