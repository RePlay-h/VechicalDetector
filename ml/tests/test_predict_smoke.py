import torch
from ml.models import build_model


def test_predict_smoke_runs() -> None:
    model = build_model("efficient_pan_af", num_classes=3, backbone_name="efficientnet_b0", pretrained_backbone=False)

    x = torch.randn(1, 3, 256, 256)
    dets = model.predict(x, score_thresh=0.99)  # high threshold => likely empty

    assert isinstance(dets, list)
    assert len(dets) == 1
    assert dets[0].boxes.ndim == 2 and dets[0].boxes.shape[1] == 4
    assert dets[0].scores.ndim == 1
    assert dets[0].labels.ndim == 1