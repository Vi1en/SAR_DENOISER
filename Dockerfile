# CPU inference image for the HTTP API (checkpoints: bind-mount or bake separately).
FROM python:3.11-slim-bookworm

WORKDIR /app

ARG SAR_GIT_SHA=
ENV SAR_GIT_SHA=${SAR_GIT_SHA}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SAR_DEVICE=cpu

COPY requirements.txt .
# One resolve: NumPy<2 and stack pins from requirements.txt, plus CPU wheels from PyTorch index.
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 8000

# Sync denoise API; use profile `api` in docker-compose for Redis-backed jobs + worker on host.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
