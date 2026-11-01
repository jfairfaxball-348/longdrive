"""Write the drum stem of a piece.

Runs inside the engine's own environment, where Demucs and torch live, so it must not
import longdrive. An hour of stems won't fit in memory, so the piece goes through in
overlapping chunks and the stem is streamed to disk.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

CHUNK_S = 120
OVERLAP_S = 8


def split_chunk(model, mix, rate: int, device: str):
    import julius
    import torch
    from demucs.apply import apply_model

    resampled = julius.resample_frac(mix, rate, model.samplerate)
    with torch.inference_mode():
        stems = apply_model(model, resampled, device=device, shifts=1, overlap=0.25, split=True)
    drums = stems[0, model.sources.index("drums")].cpu()
    return julius.resample_frac(drums, model.samplerate, rate)


def main() -> int:
    import numpy as np
    import soundfile as sf
    import torch
    from demucs.pretrained import get_model

    source_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    temporary = out_path.with_name(f"{out_path.name}.partial")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model("htdemucs")
    model.to(device).eval()
    random.seed(0)

    with sf.SoundFile(source_path) as source:
        rate = source.samplerate
        chunk, overlap = CHUNK_S * rate, OVERLAP_S * rate
        fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)[:, None]
        carry = None
        with sf.SoundFile(temporary, "w", rate, 2, "FLOAT", format="RF64") as target:
            position = 0
            while position < source.frames:
                source.seek(position)
                audio = source.read(chunk, dtype="float32", always_2d=True)
                mix = torch.from_numpy(audio.T)[None]
                try:
                    drums = split_chunk(model, mix, rate, device)
                except torch.cuda.OutOfMemoryError:
                    device = "cpu"
                    model.to(device)
                    drums = split_chunk(model, mix, rate, device)
                drums = drums.numpy().T[: audio.shape[0]]
                drums = np.pad(drums, ((0, audio.shape[0] - drums.shape[0]), (0, 0)))
                if carry is not None:
                    drums[:overlap] = carry * (1 - fade_in) + drums[:overlap] * fade_in
                is_last = position + chunk >= source.frames
                target.write(drums if is_last else drums[:-overlap])
                carry = None if is_last else drums[-overlap:]
                position += chunk - overlap
                print(f"{position / rate:.0f} s", flush=True)
    os.replace(temporary, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
