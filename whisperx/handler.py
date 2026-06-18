"""RunPod serverless handler — Phase 3: DeepFilterNet preprocessing + NeMo diarization.

Transcription and diarization are handled by the whisper-diarization subprocess
(https://github.com/MahmoudAshraf97/whisper-diarization) which combines Whisper
large-v3 with NeMo's ClusteringDiarizer instead of pyannote.

Language is always Turkish ("tr") — hardcoded throughout.

Input (job["input"]):
  audio_file     str   Presigned URL of the audio file
  model          str   Whisper model name passed to subprocess (default: "large-v3")
  min_speakers   int   Minimum number of speakers (default: 2)
  max_speakers   int   Maximum number of speakers (default: 2)
  -- ignored but accepted for backward compat --
  language       str
  diarize        bool
  hf_token       str
  batch_size     int
  compute_type   str

Output:
  {
    "segments": [
      {"id": 0, "start": 0.0, "end": 2.5, "text": "...", "speaker": "SPEAKER_00"},
      ...
    ],
    "language": "tr"
  }
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request

import noisereduce as nr
import numpy as np
import pyloudnorm as pyln
import runpod
import torch
import torchaudio

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _init_error_tracking() -> bool:
    """Point the Sentry SDK at GlitchTip when SENTRY_DSN is set. Returns whether
    capture is active. No transcript text is ever sent — only the job id and the
    exception/stack from a failed transcription."""
    dsn = os.environ.get("SENTRY_DSN")
    if not dsn:
        return False
    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=dsn,
            environment=os.environ.get("SENTRY_ENVIRONMENT", "production"),
            send_default_pii=False,
            traces_sample_rate=0.0,
        )
        sentry_sdk.set_tag("component", "transcriptor")
        return True
    except Exception as exc:
        print(f"[sentry] init failed: {exc!r}")
        return False


_SENTRY_ON = _init_error_tracking()


def _gpu_name() -> str:
    """Resolve the assigned GPU model so the backend can attribute cost.

    RunPod assigns A4000 / A5000 / A100 at the pool level; the worker is the
    only place that can see which one actually ran the job.
    """
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"


def _audio_duration_s(path: str) -> float:
    """Wall-clock length of the audio in seconds — the primary cost driver.

    Uses the lightweight torchaudio metadata probe (no full decode). Returns
    0.0 if the container doesn't report frame count; the caller falls back to
    the last segment's end time.
    """
    try:
        info = torchaudio.info(path)
        if info.num_frames and info.sample_rate:
            return info.num_frames / info.sample_rate
    except Exception:
        pass
    return 0.0


def preprocess_audio(input_path: str, output_path: str) -> str:
    """Convert to 16 kHz mono WAV, denoise with noisereduce, normalize to -23 LUFS."""
    audio, sr = torchaudio.load(input_path)

    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    target_sr = 16_000
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)

    audio_np = audio.squeeze().numpy()

    # Stationary noise reduction before diarization
    audio_np = nr.reduce_noise(
        y=audio_np, sr=target_sr, stationary=False, prop_decrease=0.75
    ).astype(np.float32)

    # Loudness normalize to -23 LUFS
    meter = pyln.Meter(target_sr)
    loudness = meter.integrated_loudness(audio_np)
    if loudness > -70:  # skip near-silence (pyln would blow up gain)
        audio_np = pyln.normalize.loudness(audio_np, loudness, -23.0)

    torchaudio.save(output_path, torch.tensor(audio_np).unsqueeze(0), target_sr)
    print(f"[preprocess] Saved cleaned audio → {output_path}")
    return output_path


# --- NeMo diarization via whisper-diarization subprocess ---------------------

def diarize_with_nemo(
    audio_path: str,
    output_dir: str,
    model: str = "large-v3",
) -> str:
    """Run whisper-diarization (NeMo backend). Returns path to the output SRT file."""
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python", "/app/whisper-diarization/diarize.py",
        "-a", audio_path,
        "--whisper-model", model,
        "--device", DEVICE,
        "--language", "tr",
        "--no-stem",  # disable Demucs source separation (audio already preprocessed)
    ]
    print(f"[nemo] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=output_dir, timeout=3600, capture_output=True, text=True)
    if result.stdout:
        print(f"[nemo] stdout:\n{result.stdout[-3000:]}")
    if result.stderr:
        print(f"[nemo] stderr:\n{result.stderr[-3000:]}")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)

    srt_path = f"{os.path.splitext(audio_path)[0]}.srt"
    if not os.path.exists(srt_path):
        raise FileNotFoundError(f"Expected SRT output not found: {srt_path}")
    return srt_path


def _normalize_speaker(raw_label: str) -> str:
    """Convert 'Speaker N' → 'SPEAKER_0N' to match backend expectations."""
    m = re.match(r"Speaker\s+(\d+)$", raw_label, re.IGNORECASE)
    if m:
        return f"SPEAKER_{int(m.group(1)):02d}"
    return raw_label


def _srt_ts_to_s(ts: str) -> float:
    """Parse SRT timestamp 'HH:MM:SS,mmm' → seconds."""
    h, m, s = ts.strip().replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_srt(srt_path: str) -> list[dict]:
    """Parse whisper-diarization SRT output into [{id, start, end, text, speaker}]."""
    with open(srt_path, encoding="utf-8") as f:
        content = f.read()

    segments = []
    for i, block in enumerate(content.strip().split("\n\n")):
        lines = [ln for ln in block.strip().splitlines() if ln.strip()]
        if len(lines) < 3 or " --> " not in lines[1]:
            continue

        start_str, end_str = lines[1].split(" --> ", 1)
        raw_text = " ".join(lines[2:]).strip()

        speaker = ""
        text = raw_text
        # whisper-diarization prefixes lines with "SPEAKER_XX: " or "Speaker N: "
        for prefix in ("SPEAKER_", "Speaker "):
            if raw_text.startswith(prefix):
                colon = raw_text.find(": ")
                if colon != -1:
                    speaker = _normalize_speaker(raw_text[:colon].strip())
                    text = raw_text[colon + 2:].strip()
                break

        segments.append({
            "id": i,
            "start": round(_srt_ts_to_s(start_str), 3),
            "end": round(_srt_ts_to_s(end_str), 3),
            "text": text,
            "speaker": speaker,
        })

    return segments


def _fill_missing_speakers(segments: list[dict]) -> None:
    """Propagate speaker labels forward to any unlabeled segments."""
    last = ""
    for seg in segments:
        if seg.get("speaker"):
            last = seg["speaker"]
        elif last:
            seg["speaker"] = last


def handler(job: dict) -> dict:
    inp = job["input"]
    audio_url: str = inp["audio_file"]
    model: str = inp.get("model", "large-v3")
    # language always "tr"; min/max_speakers, hf_token, diarize, batch_size, compute_type ignored

    suffix = ".wav"
    for ext in (".mp3", ".m4a", ".ogg", ".flac", ".aac", ".webm"):
        if ext in audio_url.lower():
            suffix = ext
            break

    tmp_path = None
    cleaned_path = None
    output_dir = None
    segments: list[dict] = []
    # Cost-tracking checkpoints. The backend turns these into a billable
    # UsageEvent (audio minutes + GPU-seconds). All times are wall-clock.
    t_start = time.time()
    t_downloaded = t_preprocessed = t_diarized = t_start
    file_bytes = 0
    audio_seconds = 0.0
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as cleaned:
            cleaned_path = cleaned.name
        output_dir = tempfile.mkdtemp()

        # 1. Download + preprocess
        print(f"[handler] Downloading audio → {tmp_path}")
        urllib.request.urlretrieve(audio_url, tmp_path)
        t_downloaded = time.time()
        try:
            file_bytes = os.path.getsize(tmp_path)
        except OSError:
            file_bytes = 0
        audio_seconds = _audio_duration_s(tmp_path)
        preprocess_audio(tmp_path, cleaned_path)
        t_preprocessed = time.time()

        # 2. Transcribe + diarize via NeMo hybrid subprocess
        srt_path = diarize_with_nemo(cleaned_path, output_dir, model)
        segments = parse_srt(srt_path)
        _fill_missing_speakers(segments)
        t_diarized = time.time()

        from collections import Counter
        dist = Counter(s.get("speaker", "") for s in segments)
        print(f"[nemo] Speaker distribution: {dict(dist)}")

    finally:
        for p in (tmp_path, cleaned_path):
            if p and os.path.exists(p):
                os.unlink(p)
        if output_dir:
            shutil.rmtree(output_dir, ignore_errors=True)

    # Fall back to the last segment's end if the container had no frame count.
    if audio_seconds <= 0.0 and segments:
        audio_seconds = max((s.get("end", 0.0) for s in segments), default=0.0)

    t_end = time.time()
    metrics = {
        "audio_seconds": round(audio_seconds, 2),
        "file_bytes": file_bytes,
        "processing_seconds": round(t_end - t_start, 2),
        "download_seconds": round(t_downloaded - t_start, 2),
        "preprocess_seconds": round(t_preprocessed - t_downloaded, 2),
        "diarize_seconds": round(t_diarized - t_preprocessed, 2),
        "gpu_name": _gpu_name(),
        "device": DEVICE,
        "model": model,
        "num_speakers": len({s.get("speaker", "") for s in segments if s.get("speaker")}),
        "num_segments": len(segments),
    }
    # Single structured line so RunPod's captured stdout is grep/JSON-parseable.
    print("[metrics] " + json.dumps(metrics))
    print(f"[nemo] Done. Output segments: {len(segments)}")
    return {"segments": segments, "language": "tr", "metrics": metrics}


def _entry(job: dict) -> dict:
    """Capture failures to GlitchTip (with the job id tag) then re-raise so
    RunPod still marks the job FAILED."""
    if _SENTRY_ON:
        import sentry_sdk

        with sentry_sdk.configure_scope() as scope:
            scope.set_tag("runpod_job_id", str(job.get("id", "")))
    try:
        return handler(job)
    except Exception:
        if _SENTRY_ON:
            import sentry_sdk

            sentry_sdk.capture_exception()
            sentry_sdk.flush(timeout=2.0)
        raise


runpod.serverless.start({"handler": _entry})
