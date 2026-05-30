# Slim image for the Knative lifecycle worker. Pure CPU; no torch.
# The host owns the GPU; this container only emits lifecycle timestamps that
# the harness correlates with the host NVML stream.
FROM python:3.11-slim AS base

WORKDIR /app

RUN pip install --no-cache-dir \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.27" \
        "httpx>=0.27" \
        "pydantic>=2.6"

COPY hasagi/worker/knative_main.py /app/hasagi/worker/knative_main.py
COPY hasagi/worker/__init__.py     /app/hasagi/worker/__init__.py
RUN mkdir -p /app/hasagi && touch /app/hasagi/__init__.py

ENV PYTHONPATH=/app
EXPOSE 8080

CMD ["python", "-m", "hasagi.worker.knative_main"]
