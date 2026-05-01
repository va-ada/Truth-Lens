# TruthLens backend — production image
# ------------------------------------------------------------------
# Build context expectation: the entire repo root, so we can copy the
# trained-model artefacts from codebase/Ml_MP/TruthLens/models/ and the
# TruthLens src/ that the backend imports at boot.
#
#   docker build -f codebase/Ml_MP/Truth/backend/Dockerfile -t truthlens-api .
#   docker run -p 7860:7860 --env-file .env truthlens-api
#
# Hugging Face Spaces "Docker SDK" expects the container to listen on
# $PORT (default 7860). Render/Railway/Fly all override $PORT too — we
# read it at runtime so the same image works everywhere.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf-cache \
    TRANSFORMERS_CACHE=/data/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/data/hf-cache \
    NLTK_DATA=/data/nltk \
    PORT=7860

# System deps:
#   libglib / libgl    — easyocr / opencv runtime
#   libgomp1           — sklearn / torch threading
#   curl               — healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Backend deps first (cached layer when only the model changes).
COPY codebase/Ml_MP/Truth/backend/requirements.txt /app/backend-requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /app/backend-requirements.txt

# spaCy model used by the preprocessor's NER masking. Pinned to the
# small English model — large/transformer variants would 4x the image.
RUN python -m spacy download en_core_web_sm

# NLTK stopwords for the lemmatizer
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/data/nltk', quiet=True); nltk.download('punkt', download_dir='/data/nltk', quiet=True); nltk.download('punkt_tab', download_dir='/data/nltk', quiet=True)"

# TruthLens ML pipeline (the backend imports its src/ + config.py + scripts)
COPY codebase/Ml_MP/TruthLens/src/         /app/truthlens/src/
COPY codebase/Ml_MP/TruthLens/scripts/     /app/truthlens/scripts/
COPY codebase/Ml_MP/TruthLens/config.py    /app/truthlens/config.py
# Mark the package importable as `truthlens.*`
RUN touch /app/truthlens/__init__.py
# Trained model artefacts. ~140 MB total.
COPY codebase/Ml_MP/TruthLens/models/      /app/truthlens/models/

# Backend code
COPY codebase/Ml_MP/Truth/backend/         /app/backend/

# Pre-cache the MiniLM sentence-transformer (~80 MB) so first request is
# fast. The model is downloaded into HF_HOME which is on the persistent
# /data volume.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# The backend's main.py expects to find TruthLens via parent-directory
# traversal. Mirror that layout under /app so existing import paths work
# without code changes.
RUN mkdir -p /app/codebase/Ml_MP \
    && ln -s /app/truthlens /app/codebase/Ml_MP/TruthLens \
    && mkdir -p /app/codebase/Ml_MP/Truth \
    && ln -s /app/backend /app/codebase/Ml_MP/Truth/backend

WORKDIR /app/backend

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-7860}/ || exit 1

# uvicorn reads $PORT so HF Spaces / Render / Railway / Fly can all rewrite
# it without changing the image. Single worker — the model is loaded into
# process memory once and is not pickle-safe to fork after spaCy / torch
# are initialised.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
