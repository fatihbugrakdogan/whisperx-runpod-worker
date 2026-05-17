# syntax=docker/dockerfile:1
# ── WhisperX RunPod Serverless Worker (models baked in) ───────────────────────
#
# Build (from the workers/ repo root, not from whisperx/):
#   docker buildx build -t fatihbugra/whisperx-worker:latest -f whisperx/Dockerfile --push .
#
# Build context = workers/ root so we can COPY both whisperx/ and the local
# whisper-diarization fork that lives at workers/diarization/.
#
# RunPod Serverless:
#   Container image : fatihbugra/whisperx-worker:latest
#   Container disk  : 40 GB
#   GPU             : A4000 / A5000 / A100
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models
ENV HF_HUB_CACHE=/models/hub
ENV TRANSFORMERS_CACHE=/models/hub

# ── System packages ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-distutils ffmpeg git wget curl \
        build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN wget -qO- https://bootstrap.pypa.io/get-pip.py | python3.11

RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ── Python dependencies ────────────────────────────────────────────────────────
COPY whisperx/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        -r /tmp/requirements.txt

# ── whisper-diarization (local fork at workers/diarization/) ──────────────────
# Replaces the previous `git clone github.com/MahmoudAshraf97/whisper-diarization`
# so our domain-specific patches ship to prod:
#   - num_speakers=2 fixed for therapy diarization
#   - condition_on_previous_text=False (Whisper hallucination guard)
#   - --diarizer {msdd,sortformer} flag for runtime backend selection
# Upstream improvements are pulled in by syncing workers/diarization/ from the
# fork; nothing here is auto-updated from GitHub.
COPY diarization /app/whisper-diarization
RUN pip install --no-cache-dir \
       -c /app/whisper-diarization/constraints.txt \
       -r /app/whisper-diarization/requirements.txt

# ── Bake Whisper large-v3 + Turkish alignment model ───────────────────────────
COPY whisperx/download_models.py /tmp/download_models.py
RUN python /tmp/download_models.py

# ── App ────────────────────────────────────────────────────────────────────────
WORKDIR /app
COPY whisperx/handler.py .

CMD ["python", "-u", "handler.py"]
