# AVT Car Bounding Box YOLO API

FastAPI service for detecting the main vehicle bounding box in regular and
360-degree panorama photos of cars, vans and trucks. The service uses YOLO for vehicle detection,
ranks candidates by visible area with a vertical-center bonus, and includes a
roll-verification pass for cars split across the left/right edge of an
equirectangular projection.

## What It Returns

The main response field is:

```json
{
  "bbox_xyxy": [5503.612548828125, 1325.8834228515625, 850.34619140625, 2058.998291015625],
  "wraps_x": true
}
```

`bbox_xyxy` is always `[xMin, yMin, xMax, yMax]` in input-image pixel
coordinates.

If `wraps_x` is `false`, then `xMin <= xMax`.

If `wraps_x` is `true`, the car crosses the horizontal panorama seam and
`xMax` may be smaller than `xMin`. In that case the bbox is interpreted as two
visible segments:

- right segment: `xMin .. image_width`
- left segment: `0 .. xMax`

This keeps coordinates in the original image geometry instead of returning a
modified/rolled image coordinate system.

## Project Layout

```text
app.py                    FastAPI app, YOLO loading, API schemas, endpoints
car_detection_logic.py    Pure bbox ranking and roll-verification logic
findCar.sh                Runpod helper script for one image
tests/                    Pure detection logic and API regression tests
requirements.txt          Runtime dependencies
requirements-test.txt     Lightweight test dependencies
requirements-dev.txt      Runtime plus test dependencies
Dockerfile                CPU-only distributable container
Dockerfile.gpu            NVIDIA CUDA container for GPU hosts
```

Local sample images and generated YOLO outputs live under `examples/`, which is
ignored by git.

## Python Environment

The repo is pinned to Python `3.10.8` for both `asdf` and `pyenv`:

```text
.tool-versions
.python-version
```

With `asdf`:

```bash
asdf install
python3 -m venv .venv
source .venv/bin/activate
```

With `pyenv`:

```bash
pyenv install 3.10.8
pyenv local 3.10.8
python -m venv .venv
source .venv/bin/activate
```

## Running Tests

Install the lightweight test dependencies (no Torch, Ultralytics, or model weights
are needed). API tests use fixed YOLO outputs to exercise both endpoints:

```bash
python -m pip install -r requirements-test.txt
python -m pytest
```

Coverage report:

```bash
python -m pytest --cov=car_detection_logic --cov-report=term-missing
```

The tests embed selected real Runpod/YOLO outputs from:

```text
examples/runpod_outputs/b_pack_v2
examples/runpod_outputs/c_pack_v2
```

The intentionally excluded cases are `b06`, `b07`, and `c06`, because those
images represent ambiguous or unusual placement cases that should not anchor the
normal edge-case expectations.

## Running Locally

Install full runtime dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

Run the API:

```bash
python app.py
```

The service listens on `0.0.0.0:8000` by default. Override with:

```bash
PORT=8080 python app.py
```

Health check:

```bash
curl http://localhost:8000/health
```

Readiness check:

```bash
curl http://localhost:8000/ping
```

## API Docs

FastAPI generates OpenAPI docs automatically:

```text
http://localhost:8000/docs
http://localhost:8000/redoc
http://localhost:8000/openapi.json
```

Export the OpenAPI schema:

```bash
curl http://localhost:8000/openapi.json > openapi.json
```

The API schemas and examples are defined in `app.py` with Pydantic models.

## Endpoints

### `GET /ping`

Returns `200` when the model has warmed up:

```json
{
  "status": "healthy"
}
```

Returns `503` while the worker is still warming:

```json
{
  "detail": "Model warming up"
}
```

### `GET /health`

Returns model and container metadata:

```json
{
  "ok": true,
  "model_ready": true,
  "app_version": "0.1.18",
  "model": "yolov8n.pt",
  "vehicle_class_ids": [2, 5, 7],
  "car_class_ids": [2, 5, 7]
}
```

### `POST /detect`

Detect the main vehicle in one uploaded image.

Example:

```bash
curl -X POST "http://localhost:8000/detect?roll_verify=true&draw_output=false" \
  -F "file=@examples/a01.jpg"
```

Useful query parameters:

```text
conf                    YOLO confidence threshold, default 0.25
imgsz                   YOLO inference image size, default 640
max_det                 Max raw detections, default 300
candidate_limit         Max suggestion boxes to draw, default 10
draw_output             Include base64 JPEG with selected bbox, default false
draw_suggestions        Draw candidate boxes when draw_output=true, default false
roll_verify             Enable panorama seam verification, default true
vertical_center_weight  Vertical-center bonus strength, default 1.0, range 0..4
```

Response:

```json
{
  "filename": "b04.jpg",
  "image_size": {
    "w": 6080,
    "h": 3040
  },
  "app_version": "0.1.18",
  "bbox_xyxy": [
    5503.612548828125,
    1325.8834228515625,
    850.34619140625,
    2058.998291015625
  ],
  "confidence": 0.8547395467758179,
  "wraps_x": true
}
```

If `draw_output=true`, the response also includes:

```json
{
  "output_image_mime_type": "image/jpeg",
  "output_image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD..."
}
```

### `POST /detect-batch`

Detect the main vehicle in multiple uploaded images.

Example:

```bash
curl -X POST "http://localhost:8000/detect-batch?roll_verify=true&chunk_size=8" \
  -F "files=@examples/b01.jpg" \
  -F "files=@examples/b02.jpg"
```

Additional query parameter:

```text
chunk_size         Number of images per YOLO batch call, default 8
```

Invalid files return per-file error items while valid files continue through
detection.

## Supported Vehicles

Vehicle detection is always enabled for every supported model category; clients
need no opt-in flag. The bundled `yolov8n.pt` model uses COCO categories:

- `car` (2)
- `bus` (5)
- `truck` (7)

COCO has no separate `van` category. Vans are eligible when detected in one of
these categories. An explicit `van` label is also accepted if a future model
provides it; class IDs are resolved from the model's names rather than assumed.
Vehicle recognition still depends on the model; accepting these categories does
not guarantee a detection for every van or truck.

All categories use the same area and vertical-center ranking, including original
inference, the panorama seam pass, batch requests and model warm-up. This also
keeps SUVs that YOLO classifies as trucks or buses when doors or tailgates are open.
People, motorcycles, bicycles and other categories remain excluded.

`/health` reports `vehicle_class_ids`. The existing `car_class_ids` field remains
as a compatibility alias with the same expanded list. Detection response fields
and coordinates are unchanged.

## Main-Car Priority

Candidates are ranked using visible bbox area and the vertical position of the
bbox center. **Horizontal position is ignored** because a 360 panorama can have
any horizontal offset, including a main car split across the seam.

For image height `H`, clamped bbox center `cy`, visible bbox area `A`, and
`vertical_center_weight = w`:

```text
vertical_center_bonus = 1 - abs(2 * cy / H - 1)
score = A * (1 + w * vertical_center_bonus)
```

The bonus ranges from 0 at the top/bottom to 1 at half the image height. At the
default weight of `1`, the area multiplier ranges from 1x to 2x. Area still
matters: a tiny centered background car cannot outweigh a car more than twice
its visible area. Confidence only breaks equal-score ties.

For example, a centered car with an area of 100,000 pixels scores 200,000. A
larger foreground car with an area of 120,000 pixels and `cy = 0.75 * H` scores
180,000, so the centered car wins.

Both `/detect` and `/detect-batch` accept `vertical_center_weight` (0 through 4).
Set it to `0` for area-only ranking, or raise it to strengthen the preference for
mid-height cars. This parameter applies equally to original and verified seam
detections. Disabling the bonus keeps the seam comparison fix described below.

This is a geometric heuristic, not identification of a particular vehicle.
The default weight is a starting point; tune it against labeled crowded-parking
photos, especially when the camera is tilted. The embedded Runpod fixtures
verify seam geometry, but contain only selected bboxes, not competing detections;
they do not measure selection accuracy on crowded parking lots.

## Roll Verification Logic

The detection flow is:

1. Run YOLO on the original image.
2. Rank car candidates by area with the vertical-center bonus. Confidence only
   breaks exact score ties.
3. If a meaningful candidate touches the left or right panorama edge, roll the
   image horizontally by half its width.
4. Run YOLO on the rolled image.
5. Map rolled detections back into the original coordinate system.
6. Keep mapped detections only if they become `wraps_x=true` and overlap a
   first-pass edge candidate strongly enough.
7. Compare verified cars with the first-pass winner using the same priority
   score and confidence tie-breaker. A seam car does not automatically replace
   the main car; exact score/confidence ties retain the first-pass selection.

This handles cases where YOLO sees only one side of a car at the panorama seam,
then sees the full car after the half-roll.

## Runpod Helper

`findCar.sh` sends one image to a Runpod endpoint, waits for `/ping`, saves the
drawn output image, and saves the response JSON.

Create `.env` locally:

```bash
RUNPOD_API_KEY=your_token
RUNPOD_BASE_URL=https://your-runpod-endpoint
RUNPOD_CANDIDATE_LIMIT=10
RUNPOD_DRAW_SUGGESTIONS=false
RUNPOD_MAX_DET=300
RUNPOD_ROLL_VERIFY=true
RUNPOD_VERTICAL_CENTER_WEIGHT=1.0
RUNPOD_WARMUP_TIMEOUT_SECONDS=180
RUNPOD_POLL_INTERVAL_SECONDS=2
RUNPOD_PING_MAX_TIME_SECONDS=10
```

Run:

```bash
zsh findCar.sh /path/to/image.jpg
```

Or choose an output image path:

```bash
zsh findCar.sh /path/to/image.jpg /path/to/output.jpg
```

The script writes response JSON next to the current working directory by
default. Override with:

```bash
RUNPOD_JSON_OUTPUT_PATH=/path/to/response.json zsh findCar.sh /path/to/image.jpg
```

## Docker

The repo now ships with two container targets:

- `Dockerfile`: CPU-only image for local and team use. Smaller and easier to run anywhere, but inference is CPU-only.
- `Dockerfile.gpu`: NVIDIA/CUDA image for GPU hosts such as Runpod.

Build the default CPU image:

```bash
docker build -t antropov/avt-car-bbox-yolo:local .
```

Run the CPU image:

```bash
docker run --rm -p 8000:8000 antropov/avt-car-bbox-yolo:local
```

The CPU image installs the official PyTorch CPU wheels, which avoids pulling the
Linux CUDA runtime packages into a team-distributable container.

Build the GPU image:

```bash
docker build -f Dockerfile.gpu -t antropov/avt-car-bbox-yolo:gpu-local .
```

Run the GPU image on an NVIDIA host:

```bash
docker run --rm --gpus all -p 8000:8000 antropov/avt-car-bbox-yolo:gpu-local
```

Build a release tag for the default CPU image:

```bash
docker build -t antropov/avt-car-bbox-yolo:0.1.18 .
```

Push:

```bash
docker push antropov/avt-car-bbox-yolo:0.1.18
docker tag antropov/avt-car-bbox-yolo:0.1.18 antropov/avt-car-bbox-yolo:latest
docker push antropov/avt-car-bbox-yolo:latest
```

Both Dockerfiles expose HTTP port `8000`.

## Versioning

The API reports its app version in both `/health` and detection responses:

```json
{
  "app_version": "0.1.18"
}
```

Update `APP_VERSION` in `app.py` when changing API behavior or deployment logic.
