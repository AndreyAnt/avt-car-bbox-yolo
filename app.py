import base64
import io
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError
from fastapi import Depends, FastAPI, UploadFile, File, Query, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from ultralytics import YOLO

from car_detection_logic import (
    _bbox_metrics,
    _candidate_payload,
    _find_seam_edge_candidates,
    _rank_car_candidates_xyxy,
    _select_verified_rolled_candidate,
)

# Choose speed vs accuracy:
# - yolov8n.pt: fastest
# - yolov8s.pt: still fast, better accuracy
APP_VERSION = "0.1.15"
MODEL_PATH = "yolov8n.pt"
MAX_BATCH_FILES = 100

app = FastAPI(
    title="AVT Car Bounding Box YOLO API",
    description=(
        "Detects the main car bounding box in regular and 360 panorama vehicle "
        "photos. Horizontal panorama seam boxes are represented with `wraps_x=true` "
        "and may return `bbox_xyxy[2] < bbox_xyxy[0]`."
    ),
    version=APP_VERSION,
)
app.state.model_ready = False

model = YOLO(MODEL_PATH)

# Find class id(s) for "car" if present
CAR_CLASS_IDS: List[int] = [i for i, name in model.names.items() if name == "car"]
# Fallback (COCO typically uses id 2 for car)
if not CAR_CLASS_IDS:
    CAR_CLASS_IDS = [2]


class ImageSizeResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "w": 6080,
                    "h": 3040,
                }
            ]
        }
    )

    w: int = Field(..., description="Image width in pixels.", examples=[6080])
    h: int = Field(..., description="Image height in pixels.", examples=[3040])


class PingResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "healthy",
                }
            ]
        }
    )

    status: str = Field(..., description="Worker readiness status.", examples=["healthy"])


class HealthResponse(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "examples": [
                {
                    "ok": True,
                    "model_ready": True,
                    "app_version": APP_VERSION,
                    "model": MODEL_PATH,
                    "car_class_ids": [2],
                }
            ]
        }
    )

    ok: bool = Field(..., description="True when the model has warmed up and the worker can serve traffic.")
    model_ready: bool = Field(..., description="True after the startup warm-up inference succeeds.")
    app_version: str = Field(..., description="Application version served by this container.", examples=[APP_VERSION])
    model: str = Field(..., description="YOLO model weights path or name.", examples=[MODEL_PATH])
    car_class_ids: List[int] = Field(..., description="YOLO class ids considered to be cars.", examples=[[2]])


class DetectionOptions(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "conf": 0.25,
                    "imgsz": 640,
                    "max_det": 300,
                    "candidate_limit": 10,
                    "draw_output": False,
                    "draw_suggestions": False,
                    "roll_verify": True,
                }
            ]
        }
    )

    conf: float = Field(
        0.25,
        ge=0.0,
        le=1.0,
        description="Minimum YOLO confidence threshold for car detections.",
        examples=[0.25],
    )
    imgsz: int = Field(
        640,
        ge=320,
        le=1920,
        description="YOLO inference image size. Larger values can improve small-car detection at higher cost.",
        examples=[640],
    )
    max_det: int = Field(
        300,
        ge=1,
        le=1000,
        description="Maximum number of raw YOLO detections to keep before app-level ranking.",
        examples=[300],
    )
    candidate_limit: int = Field(
        10,
        ge=0,
        le=100,
        description="Maximum number of suggestion boxes to draw when `draw_suggestions=true`.",
        examples=[10],
    )
    draw_output: bool = Field(
        False,
        description="When true, include a JPEG with the selected bbox drawn over the input image.",
        examples=[False],
    )
    draw_suggestions: bool = Field(
        False,
        description="When true and `draw_output=true`, draw candidate suggestion boxes as well.",
        examples=[False],
    )
    roll_verify: bool = Field(
        True,
        description=(
            "When true, rerun detection on a half-rolled panorama if first-pass "
            "detections touch the horizontal seam edges."
        ),
        examples=[True],
    )


class BatchDetectionOptions(DetectionOptions):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "conf": 0.25,
                    "imgsz": 640,
                    "max_det": 300,
                    "candidate_limit": 10,
                    "chunk_size": 8,
                    "draw_output": False,
                    "draw_suggestions": False,
                    "roll_verify": True,
                }
            ]
        }
    )

    chunk_size: int = Field(
        8,
        ge=1,
        le=32,
        description="Number of uploaded images to send to YOLO in one batch inference call.",
        examples=[8],
    )


class DetectionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "filename": "b04.jpg",
                    "image_size": {"w": 6080, "h": 3040},
                    "app_version": APP_VERSION,
                    "bbox_xyxy": [
                        5503.612548828125,
                        1325.8834228515625,
                        850.34619140625,
                        2058.998291015625,
                    ],
                    "confidence": 0.8547395467758179,
                    "wraps_x": True,
                },
                {
                    "filename": "empty-parking.jpg",
                    "image_size": {"w": 1920, "h": 960},
                    "app_version": APP_VERSION,
                    "bbox_xyxy": None,
                    "confidence": None,
                    "wraps_x": False,
                },
            ]
        }
    )

    filename: Optional[str] = Field(None, description="Original uploaded filename.", examples=["b04.jpg"])
    image_size: ImageSizeResponse = Field(..., description="Decoded input image size.")
    app_version: str = Field(..., description="Application version served by this container.", examples=[APP_VERSION])
    bbox_xyxy: Optional[List[float]] = Field(
        None,
        description=(
            "Selected main-car bbox as `[xMin, yMin, xMax, yMax]` pixel coordinates. "
            "If `wraps_x=true`, the bbox crosses the panorama seam and `xMax` may be less than `xMin`."
        ),
        examples=[[5503.612548828125, 1325.8834228515625, 850.34619140625, 2058.998291015625]],
    )
    confidence: Optional[float] = Field(
        None,
        description="YOLO confidence for the selected detection, or null when no car is detected.",
        examples=[0.8547395467758179],
    )
    wraps_x: bool = Field(..., description="True when the returned bbox crosses the horizontal panorama seam.")
    output_image_mime_type: Optional[str] = Field(
        None,
        description="MIME type for `output_image_base64`; present only when `draw_output=true`.",
        examples=["image/jpeg"],
    )
    output_image_base64: Optional[str] = Field(
        None,
        description="Base64 JPEG with detection overlays; present only when `draw_output=true`.",
        examples=["/9j/4AAQSkZJRgABAQAAAQABAAD..."],
    )


class DetectionErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "filename": "broken.jpg",
                    "error": "Invalid image file",
                }
            ]
        }
    )

    filename: Optional[str] = Field(None, description="Original uploaded filename when available.")
    error: str = Field(..., description="Per-file validation error message.")


class BatchDetectionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "count": 2,
                    "items": [
                        {
                            "filename": "b04.jpg",
                            "image_size": {"w": 6080, "h": 3040},
                            "app_version": APP_VERSION,
                            "bbox_xyxy": [
                                5503.612548828125,
                                1325.8834228515625,
                                850.34619140625,
                                2058.998291015625,
                            ],
                            "confidence": 0.8547395467758179,
                            "wraps_x": True,
                        },
                        {
                            "filename": "broken.jpg",
                            "error": "Invalid image file",
                        },
                    ],
                }
            ]
        }
    )

    count: int = Field(..., description="Number of uploaded files represented in `items`.")
    items: List[Union[DetectionResponse, DetectionErrorResponse]] = Field(
        ...,
        description="Detection results in upload order. Invalid files return an error item.",
    )


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


def _detection_options(
    conf: float = Query(
        0.25,
        ge=0.0,
        le=1.0,
        description="Minimum YOLO confidence threshold for car detections.",
        examples=[0.25],
    ),
    imgsz: int = Query(
        640,
        ge=320,
        le=1920,
        description="YOLO inference image size. Larger values can improve small-car detection at higher cost.",
        examples=[640],
    ),
    max_det: int = Query(
        300,
        ge=1,
        le=1000,
        description="Maximum number of raw YOLO detections to keep before app-level ranking.",
        examples=[300],
    ),
    candidate_limit: int = Query(
        10,
        ge=0,
        le=100,
        description="Maximum number of suggestion boxes to draw when `draw_suggestions=true`.",
        examples=[10],
    ),
    draw_output: bool = Query(
        False,
        description="When true, include a JPEG with the selected bbox drawn over the input image.",
        examples=[False],
    ),
    draw_suggestions: bool = Query(
        False,
        description="When true and `draw_output=true`, draw candidate suggestion boxes as well.",
        examples=[False],
    ),
    roll_verify: bool = Query(
        True,
        description=(
            "When true, rerun detection on a half-rolled panorama if first-pass "
            "detections touch the horizontal seam edges."
        ),
        examples=[True],
    ),
) -> DetectionOptions:
    return DetectionOptions(
        conf=conf,
        imgsz=imgsz,
        max_det=max_det,
        candidate_limit=candidate_limit,
        draw_output=draw_output,
        draw_suggestions=draw_suggestions,
        roll_verify=roll_verify,
    )


def _batch_detection_options(
    conf: float = Query(
        0.25,
        ge=0.0,
        le=1.0,
        description="Minimum YOLO confidence threshold for car detections.",
        examples=[0.25],
    ),
    imgsz: int = Query(
        640,
        ge=320,
        le=1920,
        description="YOLO inference image size. Larger values can improve small-car detection at higher cost.",
        examples=[640],
    ),
    max_det: int = Query(
        300,
        ge=1,
        le=1000,
        description="Maximum number of raw YOLO detections to keep before app-level ranking.",
        examples=[300],
    ),
    candidate_limit: int = Query(
        10,
        ge=0,
        le=100,
        description="Maximum number of suggestion boxes to draw when `draw_suggestions=true`.",
        examples=[10],
    ),
    chunk_size: int = Query(
        8,
        ge=1,
        le=32,
        description="Number of uploaded images to send to YOLO in one batch inference call.",
        examples=[8],
    ),
    draw_output: bool = Query(
        False,
        description="When true, include a JPEG with the selected bbox drawn over the input image.",
        examples=[False],
    ),
    draw_suggestions: bool = Query(
        False,
        description="When true and `draw_output=true`, draw candidate suggestion boxes as well.",
        examples=[False],
    ),
    roll_verify: bool = Query(
        True,
        description=(
            "When true, rerun detection on a half-rolled panorama if first-pass "
            "detections touch the horizontal seam edges."
        ),
        examples=[True],
    ),
) -> BatchDetectionOptions:
    return BatchDetectionOptions(
        conf=conf,
        imgsz=imgsz,
        max_det=max_det,
        candidate_limit=candidate_limit,
        chunk_size=chunk_size,
        draw_output=draw_output,
        draw_suggestions=draw_suggestions,
        roll_verify=roll_verify,
    )


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


@app.get(
    "/ping",
    response_model=PingResponse,
    tags=["Health"],
    summary="Check worker readiness",
    responses={
        503: {
            "description": "Model is still warming up.",
            "content": {
                "application/json": {
                    "example": {"detail": "Model warming up"},
                }
            },
        }
    },
)
def ping():
    if not app.state.model_ready:
        raise HTTPException(status_code=503, detail="Model warming up")
    return {"status": "healthy"}


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Return model and container health metadata",
)
def health():
    return _health_payload()


def _build_detection_response(
    filename: Optional[str],
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


@app.post(
    "/detect",
    response_model=DetectionResponse,
    response_model_exclude_unset=True,
    tags=["Detection"],
    summary="Detect the main car in one image",
    description=(
        "Accepts one uploaded image, runs YOLO car detection, ranks detections "
        "by visible area, and optionally verifies seam-wrapped cars by rerunning "
        "detection on a half-rolled panorama."
    ),
    responses={
        400: {
            "description": "The uploaded file is empty or is not a valid image.",
            "content": {
                "application/json": {
                    "examples": {
                        "empty": {"value": {"detail": "Empty file"}},
                        "invalid": {"value": {"detail": "Invalid image file"}},
                    }
                }
            },
        }
    },
)
async def detect(
    options: DetectionOptions = Depends(_detection_options),
    file: UploadFile = File(
        ...,
        description="Input vehicle image. JPEG and PNG are supported by Pillow decoding.",
    ),
):
    try:
        img_array, img_w, img_h = await _load_upload_image(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Run inference. Setting classes speeds up and reduces false positives.
    result = _predict_car_result(img_array, options.conf, options.imgsz, options.max_det)

    return _build_detection_response(
        file.filename,
        img_array,
        img_w,
        img_h,
        result,
        options.draw_output,
        options.candidate_limit,
        options.draw_suggestions,
        options.roll_verify,
        options.conf,
        options.imgsz,
        options.max_det,
    )


@app.post(
    "/detect-batch",
    response_model=BatchDetectionResponse,
    response_model_exclude_unset=True,
    tags=["Detection"],
    summary="Detect the main car in multiple images",
    description=(
        "Accepts multiple uploaded images and returns one item per file in upload "
        "order. Invalid files are reported as per-file error items while valid "
        "files continue through YOLO inference."
    ),
    responses={
        400: {
            "description": "No files were uploaded or the batch exceeded the configured limit.",
            "content": {
                "application/json": {
                    "examples": {
                        "none": {"value": {"detail": "No files uploaded"}},
                        "too_many": {"value": {"detail": f"Too many files; max {MAX_BATCH_FILES}"}},
                    }
                }
            },
        }
    },
)
async def detect_batch(
    options: BatchDetectionOptions = Depends(_batch_detection_options),
    files: List[UploadFile] = File(
        ...,
        description=f"Input vehicle images. Up to {MAX_BATCH_FILES} files are accepted.",
    ),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files; max {MAX_BATCH_FILES}",
        )

    responses = [None] * len(files)

    for start in range(0, len(files), options.chunk_size):
        chunk_files = files[start : start + options.chunk_size]
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
            conf=options.conf,
            imgsz=options.imgsz,
            max_det=options.max_det,
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
                options.draw_output,
                options.candidate_limit,
                options.draw_suggestions,
                options.roll_verify,
                options.conf,
                options.imgsz,
                options.max_det,
            )

    return {"count": len(responses), "items": responses}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
