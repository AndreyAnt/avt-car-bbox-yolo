import base64
import io
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from ultralytics import YOLO

from car_detection_logic import (
    _bbox_metrics,
    _candidate_payload,
    _find_seam_edge_candidates,
    _rank_car_candidates_xyxy,
    _select_verified_rolled_candidate,
)

app = FastAPI()
app.state.model_ready = False

# Choose speed vs accuracy:
# - yolov8n.pt: fastest
# - yolov8s.pt: still fast, better accuracy
APP_VERSION = "0.1.15"
MODEL_PATH = "yolov8n.pt"
model = YOLO(MODEL_PATH)

# Find class id(s) for "car" if present
CAR_CLASS_IDS: List[int] = [i for i, name in model.names.items() if name == "car"]
# Fallback (COCO typically uses id 2 for car)
if not CAR_CLASS_IDS:
    CAR_CLASS_IDS = [2]

MAX_BATCH_FILES = 100


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


def _result_to_payload(
    result,
    img_w: int,
    img_h: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if result.boxes is None or len(result.boxes) == 0:
        return {"bbox_xyxy": None, "confidence": None, "wraps_x": False}, []

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    ranked = _rank_car_candidates_xyxy(xyxy, confs, img_w, img_h)
    i = ranked[0][0]
    best_conf = float(confs[i])
    x1, y1, x2, y2 = map(float, xyxy[i])

    cars = [
        _candidate_payload(rank, idx, xyxy[idx], confs[idx], score, img_w, img_h)
        for rank, (idx, score) in enumerate(ranked, start=1)
    ]
    main_car = {
        "bbox_xyxy": [x1, y1, x2, y2],
        "confidence": best_conf,
        "wraps_x": _bbox_metrics(x1, y1, x2, y2, img_w, img_h)["wraps_x"],
    }

    return main_car, cars


def _predict_car_result(
    img_array: np.ndarray,
    conf: float,
    imgsz: int,
    max_det: int,
):
    results = model.predict(
        source=img_array,
        conf=conf,
        imgsz=imgsz,
        max_det=max_det,
        classes=CAR_CLASS_IDS,
        verbose=False,
    )
    return results[0]


def _rolled_verify_payload(
    img_array: np.ndarray,
    img_w: int,
    img_h: int,
    candidates: List[Dict[str, Any]],
    conf: float,
    imgsz: int,
    max_det: int,
) -> Optional[Dict[str, Any]]:
    edge_candidates = _find_seam_edge_candidates(candidates, img_w, img_h)
    if not edge_candidates:
        return None

    shift = img_w // 2
    rolled_img = np.roll(img_array, shift=shift, axis=1)
    rolled_result = _predict_car_result(rolled_img, conf, imgsz, max_det)
    _, rolled_candidates = _result_to_payload(rolled_result, img_w, img_h)
    if not rolled_candidates:
        return None

    return _select_verified_rolled_candidate(
        rolled_candidates,
        edge_candidates,
        img_w,
        img_h,
        shift,
    )


def _draw_bbox(draw: ImageDraw.ImageDraw, bbox_xyxy, img_w: int, color, line_width: int) -> None:
    x1, y1, x2, y2 = bbox_xyxy
    if x2 < x1:
        draw.rectangle((x1, y1, img_w - 1, y2), outline=color, width=line_width)
        draw.rectangle((0, y1, x2, y2), outline=color, width=line_width)
    else:
        draw.rectangle(tuple(bbox_xyxy), outline=color, width=line_width)


def _draw_bbox_image_base64(
    img_array: np.ndarray,
    bbox_xyxy,
    suggestion_bboxes: List[List[float]],
) -> str:
    image = Image.fromarray(img_array)
    img_w, _ = image.size

    if bbox_xyxy is not None or suggestion_bboxes:
        draw = ImageDraw.Draw(image)
        line_width = max(4, min(image.size) // 250)
        for suggestion_bbox in suggestion_bboxes:
            _draw_bbox(draw, suggestion_bbox, img_w, (255, 190, 0), line_width)
        if bbox_xyxy is not None:
            _draw_bbox(draw, bbox_xyxy, img_w, (255, 0, 0), line_width + 2)

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    return base64.b64encode(output.getvalue()).decode("ascii")


def _health_payload():
    return {
        "ok": app.state.model_ready,
        "model_ready": app.state.model_ready,
        "app_version": APP_VERSION,
        "model": MODEL_PATH,
        "car_class_ids": CAR_CLASS_IDS,
    }


@app.on_event("startup")
def warm_up_model():
    # Warm a tiny inference before the worker starts taking traffic. This reduces
    # first-request latency spikes and helps Runpod only route to actually-ready workers.
    warmup_image = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(
        source=warmup_image,
        conf=0.25,
        imgsz=640,
        classes=CAR_CLASS_IDS,
        verbose=False,
    )
    app.state.model_ready = True


@app.get("/ping")
def ping():
    if not app.state.model_ready:
        raise HTTPException(status_code=503, detail="Model warming up")
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
    candidate_limit: int,
    draw_suggestions: bool,
    roll_verify: bool,
    conf: float,
    imgsz: int,
    max_det: int,
):
    detection_payload, candidates = _result_to_payload(result, img_w, img_h)
    if roll_verify:
        verified_payload = _rolled_verify_payload(
            img_array,
            img_w,
            img_h,
            candidates,
            conf,
            imgsz,
            max_det,
        )
        if verified_payload is not None:
            detection_payload = verified_payload

    payload = {
        "filename": filename,
        "image_size": {"w": img_w, "h": img_h},
        "app_version": APP_VERSION,
        **detection_payload,
    }

    if draw_output:
        suggestion_bboxes = []
        if draw_suggestions:
            suggestion_bboxes = [
                car["bbox_xyxy"]
                for car in candidates[: max(0, candidate_limit)]
            ]
        payload["output_image_mime_type"] = "image/jpeg"
        payload["output_image_base64"] = _draw_bbox_image_base64(
            img_array,
            payload["bbox_xyxy"],
            suggestion_bboxes,
        )

    return payload


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    imgsz: int = Query(640, ge=320, le=1920),
    max_det: int = Query(300, ge=1, le=1000),
    candidate_limit: int = Query(10, ge=0, le=100),
    draw_output: bool = Query(False),
    draw_suggestions: bool = Query(False),
    roll_verify: bool = Query(True),
):
    try:
        img_array, img_w, img_h = await _load_upload_image(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Run inference. Setting classes speeds up and reduces false positives.
    result = _predict_car_result(img_array, conf, imgsz, max_det)

    return _build_detection_response(
        file.filename,
        img_array,
        img_w,
        img_h,
        result,
        draw_output,
        candidate_limit,
        draw_suggestions,
        roll_verify,
        conf,
        imgsz,
        max_det,
    )


@app.post("/detect-batch")
async def detect_batch(
    files: List[UploadFile] = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    imgsz: int = Query(640, ge=320, le=1920),
    max_det: int = Query(300, ge=1, le=1000),
    candidate_limit: int = Query(10, ge=0, le=100),
    chunk_size: int = Query(8, ge=1, le=32),
    draw_output: bool = Query(False),
    draw_suggestions: bool = Query(False),
    roll_verify: bool = Query(True),
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
            max_det=max_det,
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
                candidate_limit,
                draw_suggestions,
                roll_verify,
                conf,
                imgsz,
                max_det,
            )

    return {"count": len(responses), "items": responses}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
