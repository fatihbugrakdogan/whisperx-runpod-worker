"""Downloads Whisper large-v3 and Turkish alignment model into the image."""
import os
import sys

os.makedirs("/models/hub", exist_ok=True)

print("==> Downloading Whisper large-v3 (int8, CPU) ...")
from faster_whisper import WhisperModel
WhisperModel("large-v3", device="cpu", compute_type="int8")
print("==> Whisper large-v3 OK")

print("==> Downloading Turkish alignment model ...")
try:
    import whisperx
    model, _ = whisperx.load_align_model(language_code="tr", device="cpu")
    del model
    print("==> Turkish alignment model OK")
except Exception as e:
    print(f"==> Turkish alignment model unavailable: {e} (will skip at runtime)")
