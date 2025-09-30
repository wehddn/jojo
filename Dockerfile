FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc build-essential libstdc++6 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    TRANSFORMERS_CACHE=/app/.hf_cache \
    HF_HOME=/app/.hf_cache \
    MODEL_DIR=/app/model \
    UVICORN_WORKERS=2

COPY main.py ner_runtime.py /app/
COPY model/ /app/model/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

CMD ["bash", "-lc", "uvicorn main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS}"]