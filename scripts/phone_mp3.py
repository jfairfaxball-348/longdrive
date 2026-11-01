"""Split a master into 128 kbps MP3 parts, each small enough to send to a phone (30 MiB).

The parts are written next to the master:

    uv run python scripts/phone_mp3.py path/to/master.wav
"""
import math
import sys
from pathlib import Path

import soundfile as sf

PART_S = 1800
# libsndfile maps compression 0 to 1 onto 320 to 32 kbps at a constant bitrate
COMPRESSION = (320 - 128) / (320 - 32) - 0.0005

wav = Path(sys.argv[1])
with sf.SoundFile(wav) as source:
    sr = source.samplerate
    part_frames = PART_S * sr
    # An hour often runs a few frames over, which shouldn't earn a part of its own.
    parts = max(1, math.ceil((source.frames - sr) / part_frames))
    for part in range(parts):
        path = wav.with_name(f"{wav.stem}-part{part + 1}.mp3" if parts > 1 else f"{wav.stem}.mp3")
        if path.exists():
            raise SystemExit(f"already exists: {path}")
        start = part * part_frames
        left = (source.frames if part == parts - 1 else start + part_frames) - start
        source.seek(start)
        with sf.SoundFile(
            path, "w", sr, 2, format="MP3", subtype="MPEG_LAYER_III",
            bitrate_mode="CONSTANT", compression_level=COMPRESSION,
        ) as target:
            while left > 0:
                block = source.read(min(left, sr * 10), dtype="float32", always_2d=True)
                target.write(block)
                left -= len(block)
        print(path, f"{path.stat().st_size / 2**20:.1f} MiB")
