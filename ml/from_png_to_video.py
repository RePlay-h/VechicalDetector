from pathlib import Path
import cv2

frames_dir = Path("data/demo_video")
out_path = Path("data/demo_video/demo.mp4")
fps = 30

frame_paths = sorted(frames_dir.glob("*.jpg"))

first = cv2.imread(str(frame_paths[0]))
h, w = first.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

for p in frame_paths:
    img = cv2.imread(str(p))
    if img is None:
        continue
    if img.shape[1] != w or img.shape[0] != h:
        img = cv2.resize(img, (w, h))
    writer.write(img)

writer.release()
print("saved:", out_path)