"""RunPod serverless handler for WhisperX transcription + diarization.

Input (job["input"]):
  audio_file     str   Presigned URL of the audio file
  model          str   Whisper model name (default: "large-v3")
  language       str   Language code (default: "tr")
  diarize        bool  Enable speaker diarization (default: true)
  hf_token       str   HuggingFace token for pyannote diarization model
  min_speakers   int   Minimum number of speakers (default: 2)
  max_speakers   int   Maximum number of speakers (default: 2)
  batch_size     int   Transcription batch size (default: 16)
  compute_type   str   "float16" (GPU) or "int8" (CPU fallback) (default: "float16")

Output:
  {
    "segments": [
      {"id": 0, "start": 0.0, "end": 2.5, "text": "...", "speaker": "SPEAKER_00"},
      ...
    ],
    "language": "tr"
  }
"""

import gc
import os
import tempfile
import urllib.request

import runpod
import torch
import whisperx

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Models are baked into the image at /models (set via ENV in Dockerfile).
# HF_HOME, TRANSFORMERS_CACHE, HF_HUB_CACHE are already correct — no override.

# --- Global warm-start caches -------------------------------------------------
_whisper_model = None
_whisper_model_key: tuple | None = None  # (model_name, compute_type)

_align_cache: dict[str, tuple | None] = {}  # language → (model, metadata) or None

_diarize_pipeline = None
_diarize_hf_token: str | None = None


def _get_whisper_model(model_name: str, compute_type: str):
    global _whisper_model, _whisper_model_key
    key = (model_name, compute_type)
    if _whisper_model_key != key:
        if _whisper_model is not None:
            del _whisper_model
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        print(f"[whisperx] Loading model {model_name} ({compute_type}) on {DEVICE}")
        _whisper_model = whisperx.load_model(model_name, DEVICE, compute_type=compute_type)
        _whisper_model_key = key
    return _whisper_model


def _get_align_model(language: str):
    """Returns (align_model, metadata) or None if unavailable for the language."""
    global _align_cache
    if language not in _align_cache:
        try:
            print(f"[whisperx] Loading alignment model for '{language}'")
            model, metadata = whisperx.load_align_model(
                language_code=language, device=DEVICE
            )
            _align_cache[language] = (model, metadata)
        except Exception as exc:
            print(f"[whisperx] No alignment model for '{language}': {exc} — skipping align step")
            _align_cache[language] = None
    return _align_cache[language]


def _get_diarize_pipeline(hf_token: str):
    global _diarize_pipeline, _diarize_hf_token
    if _diarize_hf_token != hf_token or _diarize_pipeline is None:
        print("[whisperx] Loading diarization pipeline")
        _diarize_pipeline = whisperx.DiarizationPipeline(
            use_auth_token=hf_token, device=DEVICE
        )
        _diarize_hf_token = hf_token
    return _diarize_pipeline


def handler(job: dict) -> dict:
    inp = job["input"]
    audio_url: str = inp["audio_file"]
    model_name: str = inp.get("model", "large-v3")
    language: str = inp.get("language", "tr")
    do_diarize: bool = bool(inp.get("diarize", True))
    hf_token: str = inp.get("hf_token", "")
    min_speakers: int = int(inp.get("min_speakers", 2))
    max_speakers: int = int(inp.get("max_speakers", 2))
    batch_size: int = int(inp.get("batch_size", 16))
    compute_type: str = inp.get("compute_type", "float16" if DEVICE == "cuda" else "int8")

    # 1. Download audio ---------------------------------------------------------
    suffix = ".wav"
    for ext in (".mp3", ".m4a", ".ogg", ".flac", ".aac", ".webm"):
        if ext in audio_url.lower():
            suffix = ext
            break

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
        print(f"[whisperx] Downloading audio from URL → {tmp_path}")
        urllib.request.urlretrieve(audio_url, tmp_path)
        audio = whisperx.load_audio(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # 2. Transcribe -------------------------------------------------------------
    print(f"[whisperx] Transcribing (model={model_name}, lang={language}, batch={batch_size})")
    model = _get_whisper_model(model_name, compute_type)
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    detected_language = result.get("language", language)
    print(f"[whisperx] Detected language: {detected_language}, segments: {len(result['segments'])}")

    # 3. Align (word-level timestamps, best effort) -----------------------------
    align_result = _get_align_model(detected_language)
    if align_result is not None:
        align_model, metadata = align_result
        print("[whisperx] Aligning…")
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            DEVICE,
            return_char_alignments=False,
        )

    # 4. Diarize (speaker labels) -----------------------------------------------
    if do_diarize and hf_token:
        print(f"[whisperx] Diarizing (min={min_speakers}, max={max_speakers})…")
        diarize = _get_diarize_pipeline(hf_token)
        diarize_segments = diarize(
            audio,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        result = whisperx.assign_word_speakers(diarize_segments, result)
    elif do_diarize and not hf_token:
        print("[whisperx] WARNING: diarize=true but hf_token is empty — skipping diarization")

    # 5. Normalise output -------------------------------------------------------
    out_segments = []
    for i, seg in enumerate(result.get("segments", [])):
        out_segments.append(
            {
                "id": i,
                "start": round(float(seg.get("start", 0.0)), 3),
                "end": round(float(seg.get("end", 0.0)), 3),
                "text": seg.get("text", "").strip(),
                "speaker": seg.get("speaker", ""),
            }
        )

    print(f"[whisperx] Done. Output segments: {len(out_segments)}")
    return {
        "segments": out_segments,
        "language": detected_language,
    }


runpod.serverless.start({"handler": handler})
