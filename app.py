import io
import math
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query
from ultralytics import YOLO

app = FastAPI()

# Choose speed vs accuracy:
# - yolov8n.pt: fastest
# - yolov8s.pt: still fast, better accuracy
MODEL_PATH = "yolov8n.pt"
model = YOLO(MODEL_PATH)

# Find class id(s) for "car" if present
CAR_CLASS_IDS: List[int] = [i for i, name in model.names.items() if name == "car"]
# Fallback (COCO typically uses id 2 for car)
if not CAR_CLASS_IDS:
    CAR_CLASS_IDS = [2]


def _pick_main_car_xyxy(
    boxes_xyxy: np.ndarray,
    confs: np.ndarray,
    img_w: int,
    img_h: int,
) -> Tuple[int, float]:
    """Pick the main car: largest area, tie-break by closeness to image center."""
    cx_img, cy_img = img_w / 2.0, img_h / 2.0
    img_area = img_w * img_h

    best_i = 0
    best_score = -1e30

    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        area = w * h

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dist = math.hypot((cx - cx_img) / img_w, (cy - cy_img) / img_h)

        # Area dominates; center is a mild tie-breaker.
        score = area - (0.15 * img_area) * dist

        if score > best_score:
            best_score = score
            best_i = i

    return best_i, float(confs[best_i])


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_PATH, "car_class_ids": CAR_CLASS_IDS}


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    imgsz: int = Query(640, ge=320, le=1920),
):
    # Read bytes → PIL
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img_w, img_h = img.size

    # Run inference. Setting classes speeds up and reduces false positives.
    results = model.predict(
        source=np.array(img),
        conf=conf,
        imgsz=imgsz,
        classes=CAR_CLASS_IDS,
        verbose=False,
    )
    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        return {"bbox_xyxy": None, "confidence": None}

    xyxy = r.boxes.xyxy.cpu().numpy()  # pixel coords
    confs = r.boxes.conf.cpu().numpy()

    i, best_conf = _pick_main_car_xyxy(xyxy, confs, img_w, img_h)
    x1, y1, x2, y2 = map(float, xyxy[i])

    return {
        "image_size": {"w": img_w, "h": img_h},
        "bbox_xyxy": [x1, y1, x2, y2],
        "confidence": best_conf,
    }
