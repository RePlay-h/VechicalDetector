import torch
from ml.models import build_model

def test_forward_shapes():
    model = build_model("efficient_pan_af", num_classes=7, 
                        backbone_name="efficientnet_b0", pretrained_backbone=False)
    model.eval()

    x = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        out = model(x)

    assert len(out.cls_logits) == 3
    assert len(out.box_reg) == 3
    assert out.centerness is not None and len(out.centerness) == 3

    for i in range(3):
        assert out.cls_logits[i].shape[0] == 2
        assert out.cls_logits[i].shape[1] == 7
        assert out.box_reg[i].shape[1] == 4
        assert out.centerness[i].shape[1] == 1