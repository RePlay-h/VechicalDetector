from pydantic import BaseModel
from typing import List

class Detection(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str

class DetectResponse(BaseModel):
    model: str
    detections: List[Detection]