FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    MPLBACKEND=Agg \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    PORT=8000 \
    PORT_HEALTH=8000
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN grep -v '^ultralytics==' requirements.txt > requirements.runtime.txt \
    && python3 -m pip install --no-cache-dir -r requirements.runtime.txt \
    && python3 -m pip install --no-cache-dir --no-deps ultralytics==8.3.0 \
    && rm requirements.runtime.txt

COPY app.py car_detection_logic.py .

# Optional: pre-download weights at build time (faster first request).
# If your build environment has no internet, comment this out and YOLO will download at runtime.
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

EXPOSE 8000 80
CMD ["python3", "app.py"]
