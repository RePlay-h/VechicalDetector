from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from loguru import logger
import numpy as np
import cv2

from .config import MODEL_REGISTRY
from .schemas import DetectResponse, Detection
from .utils_draw import draw_boxes
from .detectors.yolo_det import YoloDetector
from .detectors.custom_det import CustomFCOSDetector

app = FastAPI(title="VehicleDetector API")
_DETECTORS = {}

def get_detector(model_name: str):
    if model_name in _DETECTORS:
        return _DETECTORS[model_name]

    if model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")
    spec = MODEL_REGISTRY[model_name]
    logger.info(f"Loading model={model_name} kind={spec.kind} weights={spec.weights}")

    if spec.kind == "yolo":
        det = YoloDetector(str(spec.weights))
    else:
        det = CustomFCOSDetector(str(spec.weights), device="cuda:0")

    _DETECTORS[model_name] = det
    return det


@app.get("/models")
def list_models():
    return {"models": list(MODEL_REGISTRY.keys())}

@app.post("/detect/image", response_model=DetectResponse)
async def detect_image(
    model: str = Form(...),
    score_thr: float = Form(0.25),
    iou_thr: float = Form(0.6),
    file: UploadFile = File(...),
):
    data = await file.read()
    img_arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    det = get_detector(model)
    dets = det.predict(img, score_thr=score_thr, iou_thr=iou_thr)

    resp = DetectResponse(
        model=model,
        detections=[Detection(x1=x1,y1=y1,x2=x2,y2=y2,score=s,label=c) for x1,y1,x2,y2,s,c in dets]
    )
    return resp


@app.post("/detect/image_render")
async def detect_image_render(
    model: str = Form(...),
    score_thr: float = Form(0.25),
    iou_thr: float = Form(0.6),
    file: UploadFile = File(...),
):
    data = await file.read()
    img_arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    det = get_detector(model)
    dets = det.predict(img, score_thr=score_thr, iou_thr=iou_thr)

    rendered = draw_boxes(img, dets)
    ok, buf = cv2.imencode(".png", rendered)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image")
    return Response(content=buf.tobytes(), media_type="image/png")