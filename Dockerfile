FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TITRATE_MODEL_PATH=/app/artifacts/torch_surrogate.pt

ARG MODEL_SAMPLES=192
ARG MODEL_EPOCHS=80

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY experiments ./experiments

RUN python -m pip install --upgrade pip && \
    pip install ".[api]"

COPY artifacts ./artifacts

# Bake a reproducible model into the image so /health is healthy immediately.
# Production teams can replace this artifact with a separately versioned model.
RUN python experiments/train_torch_surrogate.py \
    --samples "${MODEL_SAMPLES}" \
    --epochs "${MODEL_EPOCHS}" \
    --artifact "${TITRATE_MODEL_PATH}" \
    --metrics /tmp/torch_surrogate_metrics.json && \
    useradd --create-home --uid 10001 titrate && \
    chown -R titrate:titrate /app

USER titrate

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import json, urllib.request; data=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)); assert data['status'] == 'ok'"

CMD ["uvicorn", "titrate.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
