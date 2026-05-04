# syntax=docker/dockerfile:1
# ── WhisperX RunPod Serverless Worker (models baked in) ───────────────────────
#
# Build:
#   export HF_TOKEN=hf_xxxx
#   docker buildx build --secret id=hf_token,env=HF_TOKEN \
#                        -t whisperx-worker:latest .
#
# RunPod Serverless:
#   Container image : whisperx-worker:latest
#   Container disk  : 30 GB
#   Network volume  : not needed (all models inside image)
#   GPU             : RTX 3090 / A100
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
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN wget -qO- https://bootstrap.pypa.io/get-pip.py | python3.11

RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ── Python dependencies (torch pulled in by whisperx via cu121 index) ─────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        -r /tmp/requirements.txt

# ── Bake Whisper large-v3 + Turkish alignment model ───────────────────────────
COPY download_models.py /tmp/download_models.py
RUN python /tmp/download_models.py

# ── Bake pyannote diarization model (HF token stays out of image layers) ──────
COPY download_pyannote.py /tmp/download_pyannote.py
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token 2>/dev/null || true) \
    python /tmp/download_pyannote.py

# ── App ────────────────────────────────────────────────────────────────────────
WORKDIR /app
COPY handler.py .

CMD ["python", "-u", "handler.py"]
