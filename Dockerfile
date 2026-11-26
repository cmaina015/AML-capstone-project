# Breast histopathology classifier: Next.js frontend (static export) + FastAPI + Swin Tiny + Grad-CAM.
# One container, one port: FastAPI serves /api/* and the compiled site. CPU only.

# ---------- stage 1: build the Next.js frontend to static files ----------
FROM node:24-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
# Server mode: the Analyze page posts images to FastAPI (/api/predict) instead of running ONNX in the browser.
ENV NEXT_PUBLIC_INFERENCE=server
RUN npm run build

# ---------- stage 2: Python inference server ----------
# Python 3.13 matches the Colab training environment recorded in model_config.json.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Hugging Face Spaces runs containers as UID 1000; everything writable lives under /home/user.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    ARTIFACT_DIR=/home/user/app/deployment_artifacts \
    WEB_DIR=/home/user/app/web/out \
    PORT=7860
WORKDIR /home/user/app

# CPU-only torch/torchvision come from the PyTorch index (see requirements.txt); opencv is headless,
# so no libGL/libglib system packages are needed.
COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt

COPY --chown=user app ./app
# JSON/PNG artifacts and the safetensors weights are copied. If the weights are missing, the app
# downloads them from HF_MODEL_REPO at startup instead.
COPY --chown=user deployment_artifacts ./deployment_artifacts
COPY --from=web --chown=user /web/out ./web/out

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
