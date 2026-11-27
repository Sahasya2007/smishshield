FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Create non-root user for production security
RUN adduser --disabled-password --gecos "" appuser

# Install lean inference dependencies first (cached Docker layer)
COPY requirements-inference.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements-inference.txt

# Copy inference engine, API, and ONNX model artifacts
COPY ml/__init__.py ml/
COPY ml/infer.py ml/
COPY ml/models/smish_model/onnx/ ml/models/smish_model/onnx/
COPY apps/ apps/
COPY ml/heuristics.py ml/


RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live')" || exit 1

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]