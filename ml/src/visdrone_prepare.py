
from pathlib import Path
import yaml

from pathlib import Path

from PIL import Image
from tqdm import tqdm


def convert_box(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

# Convert all labels to yolo format
def convert2yolo(input_dir: Path, output_dir: Path, img_dir: Path) -> None:

        output_dir.mkdir(parents=True, exist_ok=True)
        pbar = tqdm((input_dir).glob('*.txt'), desc=f'Converting {input_dir}')

        for f in pbar:
            img_size = Image.open((img_dir / f.name).with_suffix('.jpg')).size
            lines = []

            with open(f, 'r') as file:
                for row in [x.split(',') for x in file.read().strip().splitlines()]:
                    if row[4] == '0':
                        continue
                    cls = int(row[5]) - 1

                    # don't process pedestrian, people, bicycle classes
                    if cls < 3:
                        continue
                    box = convert_box(img_size, tuple(map(int, row[:4])))
                    lines.append(f"{cls-3} {' '.join(f"{x:.6f}" for x in box)}\n")

                # if image doesn't have any necessary classes we won't save empty label-file 
                if len(lines) != 0:
                    with open((output_dir / f.name).with_suffix('.txt'), 'w') as fl:
                        fl.writelines(lines)

# Convert train and validation labels to YOLO format
def preprare_visdrone_dataset() -> None:
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)
        train_config = params['train']

        train_labels = Path(train_config['train_labels'])
        val_labels = Path(train_config['val_labels'])
        train_yolo_dir = Path(train_config['yolo_train_labels'])
        val_yolo_dir = Path(train_config['yolo_val_labels'])
        train_img = Path(train_config['train_images'])
        val_img = Path(train_config['val_images'])
        
        convert2yolo(train_labels, train_yolo_dir, train_img)
        convert2yolo(val_labels, val_yolo_dir, val_img)

    
if __name__ == "__main__":
    preprare_visdrone_dataset()