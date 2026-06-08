FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY tare/ ./tare/

# CPU-only torch keeps the smoke container small. For real training swap to a
# CUDA base image (e.g. nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04) and install
# the matching torch wheel.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.3.0+cpu torchvision==0.18.0+cpu \
 && pip install --no-cache-dir -e .

CMD ["python", "-m", "tare.worker.main"]
