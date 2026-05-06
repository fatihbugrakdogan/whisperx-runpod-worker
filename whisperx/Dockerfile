# syntax=docker/dockerfile:1
# ── WhisperX RunPod Serverless Worker (models baked in) ───────────────────────
#
# Build:
#   docker buildx build -t fatihbugra/whisperx-worker:latest --push .
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
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        -r /tmp/requirements.txt

# ── whisper-diarization (NeMo backend) ────────────────────────────────────────
RUN git clone https://github.com/MahmoudAshraf97/whisper-diarization.git /app/whisper-diarization \
    && pip install --no-cache-dir \
       -c /app/whisper-diarization/constraints.txt \
       -r /app/whisper-diarization/requirements.txt

# ── Bake Whisper large-v3 + Turkish alignment model ───────────────────────────
COPY download_models.py /tmp/download_models.py
RUN python /tmp/download_models.py

# ── App ────────────────────────────────────────────────────────────────────────
WORKDIR /app
COPY handler.py .

CMD ["python", "-u", "handler.py"]
