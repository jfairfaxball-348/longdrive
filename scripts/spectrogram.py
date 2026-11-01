"""A one-page spectrogram of a whole piece, streamed in blocks so that an hour fits in memory.

This needs matplotlib, which is in the engine's environment rather than the project's:

    engines/ace-step/.venv/Scripts/python.exe scripts/spectrogram.py \
        runs/<name> out.png [master.wav]
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

BLOCK_S = 2.0
FFT = 8192
BANDS = 140

run = Path(sys.argv[1])
out = sys.argv[2]
wav = Path(sys.argv[3]) if len(sys.argv) > 3 else run / f"{run.name}.wav"
plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))

with sf.SoundFile(wav) as f:
    sr = f.samplerate
    freqs = np.fft.rfftfreq(FFT, 1 / sr)
    edges = np.geomspace(30, 16000, BANDS + 1)
    index = np.digitize(freqs, edges) - 1
    keep = (index >= 0) & (index < BANDS)
    counts = np.bincount(index[keep], minlength=BANDS) + 1e-9
    window = np.hanning(FFT)
    columns = []
    while True:
        x = f.read(int(BLOCK_S * sr), dtype="float32", always_2d=True)
        if len(x) < FFT:
            break
        mono = x.mean(axis=1)
        power = np.zeros(BANDS)
        frames = 0
        for start in range(0, len(mono) - FFT, FFT // 2):
            spectrum = np.abs(np.fft.rfft(mono[start : start + FFT] * window)) ** 2
            power += np.bincount(index[keep], weights=spectrum[keep], minlength=BANDS)
            frames += 1
        columns.append(10 * np.log10(power / counts / frames + 1e-12))

image = np.array(columns).T
minutes = np.arange(image.shape[1] + 1) * BLOCK_S / 60
fig, ax = plt.subplots(figsize=(18, 6))
top, bottom = np.percentile(image, 99.5), np.percentile(image, 15)
ax.pcolormesh(
    minutes, edges, image, vmin=bottom, vmax=top, cmap="magma", shading="flat", rasterized=True
)
ax.set_yscale("log")
ax.set_ylim(30, 16000)
ax.set_ylabel("Hz")
ax.set_xlabel("minutes")
for s in plan["segments"]:
    if s["fresh"]:
        ax.axvline(s["start_s"] / 60, color="cyan", lw=0.8)
ax.set_title(f"{run.name}: cyan lines are new movements")
fig.tight_layout()
fig.savefig(out, dpi=80)
print("wrote", out, image.shape)
