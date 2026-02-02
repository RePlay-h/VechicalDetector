from typing import List, Tuple
import cv2
import numpy as np



def draw_boxes(
    image_bgr: np.ndarray,
    dets: List[Tuple[float,float,float,float,float,int]],
) -> np.ndarray:
    img = image_bgr.copy()
    for x1,y1,x2,y2,score,label in dets:
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{label}:{score:.2f}",
            (int(x1), max(0, int(y1)-5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return img