import base64
import io
import math
import os
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
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

MAX_BATCH_FILES = 100


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


def _decode_image_bytes(data: bytes) -> Tuple[np.ndarray, int, int]:
    if not data:
        raise ValueError("Empty file")

    try:
        with Image.open(io.BytesIO(data)) as img:
            rgb_img = img.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Invalid image file") from exc

    img_w, img_h = rgb_img.size
    return np.array(rgb_img), img_w, img_h


async def _load_upload_image(upload: UploadFile) -> Tuple[np.ndarray, int, int]:
    return _decode_image_bytes(await upload.read())


def _result_to_payload(result, img_w: int, img_h: int):
    if result.boxes is None or len(result.boxes) == 0:
        return {"bbox_xyxy": None, "confidence": None}

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    i, best_conf = _pick_main_car_xyxy(xyxy, confs, img_w, img_h)
    x1, y1, x2, y2 = map(float, xyxy[i])

    return {
        "bbox_xyxy": [x1, y1, x2, y2],
        "confidence": best_conf,
    }


def _draw_bbox_image_base64(img_array: np.ndarray, bbox_xyxy) -> str:
    image = Image.fromarray(img_array)

    if bbox_xyxy is not None:
        draw = ImageDraw.Draw(image)
        line_width = max(4, min(image.size) // 250)
        draw.rectangle(tuple(bbox_xyxy), outline=(255, 0, 0), width=line_width)

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    return base64.b64encode(output.getvalue()).decode("ascii")


def _health_payload():
    return {"ok": True, "model": MODEL_PATH, "car_class_ids": CAR_CLASS_IDS}


@app.get("/ping")
def ping():
    return {"status": "healthy"}


@app.get("/health")
def health():
    return _health_payload()


def _build_detection_response(
    filename: str,
    img_array: np.ndarray,
    img_w: int,
    img_h: int,
    result,
    draw_output: bool,
):
    payload = {
        "filename": filename,
        "image_size": {"w": img_w, "h": img_h},
        **_result_to_payload(result, img_w, img_h),
    }

    if draw_output:
        payload["output_image_mime_type"] = "image/jpeg"
        payload["output_image_base64"] = _draw_bbox_image_base64(
            img_array,
            payload["bbox_xyxy"],
        )

    return payload


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    imgsz: int = Query(640, ge=320, le=1920),
    draw_output: bool = Query(False),
):
    try:
        img_array, img_w, img_h = await _load_upload_image(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Run inference. Setting classes speeds up and reduces false positives.
    results = model.predict(
        source=img_array,
        conf=conf,
        imgsz=imgsz,
        classes=CAR_CLASS_IDS,
        verbose=False,
    )

    return _build_detection_response(
        file.filename,
        img_array,
        img_w,
        img_h,
        results[0],
        draw_output,
    )


@app.post("/detect-batch")
async def detect_batch(
    files: List[UploadFile] = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    imgsz: int = Query(640, ge=320, le=1920),
    chunk_size: int = Query(8, ge=1, le=32),
    draw_output: bool = Query(False),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files; max {MAX_BATCH_FILES}",
        )

    responses = [None] * len(files)

    for start in range(0, len(files), chunk_size):
        chunk_files = files[start : start + chunk_size]
        chunk_arrays = []
        chunk_meta = []

        for idx, upload in enumerate(chunk_files, start=start):
            try:
                img_array, img_w, img_h = await _load_upload_image(upload)
            except ValueError as exc:
                responses[idx] = {
                    "filename": upload.filename,
                    "error": str(exc),
                }
                continue

            chunk_arrays.append(img_array)
            chunk_meta.append(
                {
                    "response_index": idx,
                    "filename": upload.filename,
                    "img_array": img_array,
                    "w": img_w,
                    "h": img_h,
                }
            )

        if not chunk_arrays:
            continue

        results = model.predict(
            source=chunk_arrays,
            conf=conf,
            imgsz=imgsz,
            classes=CAR_CLASS_IDS,
            verbose=False,
        )

        for meta, result in zip(chunk_meta, results):
            responses[meta["response_index"]] = _build_detection_response(
                meta["filename"],
                meta["img_array"],
                meta["w"],
                meta["h"],
                result,
                draw_output,
            )

    return {"count": len(responses), "items": responses}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
