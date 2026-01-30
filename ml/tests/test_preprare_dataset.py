from pathlib import Path
from ml.src.visdrone_prepare import convert2yolo
from PIL import Image

def _write_jpg(path: Path, size=(100, 100)) -> None:    
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', size).save(path, format='JPEG')

def test_convert2yolo_writes_label_when_has_class_ge_3(tmp_path: Path) -> None:
    ann_dir = tmp_path / 'raw_labels'
    img_dir = tmp_path / 'images'
    out_dir = tmp_path / 'yolo_labels'

    _write_jpg(img_dir / 'img1.jpg', size=(100, 200))
    ann_dir.mkdir(parents=True, exist_ok=True)

    (ann_dir / 'img1.txt').write_text("10,20,30,40,1,4,0,0\n")  

    convert2yolo(ann_dir, out_dir, img_dir)
    out_file = out_dir / 'img1.txt'
    
    assert out_file.exists(), "Output label file was not created"
    assert out_file.read_text().isspace() is False, "Output label file is empty"

def test_convert2yolo_does_not_write_empty_label_file(tmp_path: Path) -> None:
    ann_dir = tmp_path / 'raw_labels'
    img_dir = tmp_path / 'images'
    out_dir = tmp_path / 'yolo_labels'

    _write_jpg(img_dir / 'img2.jpg', size=(100, 200))
    ann_dir.mkdir(parents=True, exist_ok=True)

    (ann_dir / 'img2.txt').write_text("10,20,30,40,1,2,0,0\n")

    convert2yolo(ann_dir, out_dir, img_dir)

    assert not (out_dir / 'img2.txt').exists(), "Empty label file should not be created"




