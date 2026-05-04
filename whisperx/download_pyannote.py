"""Downloads pyannote/speaker-diarization-3.1 using HF_TOKEN from env."""
import os
import sys

hf_token = os.environ.get("HF_TOKEN", "").strip()
if not hf_token:
    print("==> No HF_TOKEN provided — pyannote model will download at first cold start")
    sys.exit(0)

print("==> Downloading pyannote/speaker-diarization-3.1 ...")
from pyannote.audio import Pipeline
Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
print("==> pyannote diarization model OK")
