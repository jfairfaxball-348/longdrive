"""Movement by movement: does a finished piece really change, and where might it stumble?

    uv run python scripts/review_run.py runs/<name> [master.wav]
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from longdrive import audio

NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
FFT = 8192
# Measured from here, to leave out the blends and the fades at either end of the piece.
AFTER_BLEND_S = 70
BEFORE_BLEND_S = 10
AFTER_START_S = 12
BEFORE_END_S = 24

run = Path(sys.argv[1])
wav = Path(sys.argv[2]) if len(sys.argv) > 2 else run / f"{run.name}.wav"
plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
state = json.loads((run / "state.json").read_text(encoding="utf-8"))
segments = plan["segments"]
done = state["segments"]
candidates = state["candidates"]


def clock(s):
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


def chroma(mono, sr):
    freqs = np.fft.rfftfreq(FFT, 1 / sr)
    use = (freqs >= 80) & (freqs <= 2000)
    pitch = np.round(12 * np.log2(freqs[use] / 440.0)).astype(int) + 9
    out = np.zeros(12)
    window = np.hanning(FFT)
    for start in range(0, len(mono) - FFT, FFT):
        power = np.abs(np.fft.rfft(mono[start : start + FFT] * window))[use] ** 2
        np.add.at(out, pitch % 12, power)
    return out / (np.linalg.norm(out) + 1e-9)


def key_of(c):
    ranked = []
    for root in range(12):
        for profile, mode in ((MAJOR, "maj"), (MINOR, "min")):
            r = float(np.corrcoef(c, np.roll(profile, root))[0, 1])
            ranked.append((r, f"{NAMES[root]} {mode}"))
    ranked.sort(reverse=True)
    return ranked[:2]


def tempo(mono, sr, bpm):
    """Tempo from low-end onsets, read off the autocorrelation peak near the planned tempo."""
    step = 48
    usable = len(mono) - len(mono) % step
    env = np.abs(mono[:usable]).reshape(-1, step).mean(axis=1)
    onset = np.clip(np.diff(env), 0, None)
    onset -= onset.mean()
    spectrum = np.fft.rfft(onset, 2 * len(onset))
    ac = np.fft.irfft(spectrum * np.conj(spectrum))[: len(onset)]
    rate = sr / step
    # eight beats out, for sub-0.1 BPM resolution
    lags = np.arange(int(8 * 60 / (bpm * 1.05) * rate), int(8 * 60 / (bpm * 0.95) * rate))
    best = lags[int(np.argmax(ac[lags]))]
    return 8 * 60 / (best / rate)


print("--- movements")
starts = [s for s in segments if s["index"] == 0 or s["fresh"]]
ends = [s["start_s"] for s in starts[1:]] + [plan["duration_s"]]
shapes, chromas = [], []
rows = []
for i, (s, end) in enumerate(zip(starts, ends, strict=True)):
    a = s["start_s"] + (AFTER_BLEND_S if i else AFTER_START_S)
    b = end - (BEFORE_BLEND_S if i < len(starts) - 1 else BEFORE_END_S)
    x, sr = audio.read_region(wav, a, b)
    mono = x.mean(axis=1)
    shapes.append(audio.spectral_shape(x, sr))
    chromas.append(chroma(mono, sr))
    rows.append((i, s, audio.busyness(x, sr), tempo(mono, sr, plan["bpm"]), key_of(chromas[-1])))
for i, s, busy, bpm, keys in rows:
    tone = float(np.abs(shapes[i] - shapes[0]).mean())
    step = float(np.abs(shapes[i] - shapes[i - 1]).mean()) if i else 0.0
    print(
        f"{i + 1} {clock(s['start_s'])} asked {s['key']:8s} "
        f"heard {keys[0][1]:7s} ({keys[0][0]:.2f}) "
        f"/ {keys[1][1]:7s} ({keys[1][0]:.2f}) busy {busy:4.2f} bpm {bpm:6.2f} "
        f"tone vs 1st {tone:4.1f} step {step:4.1f}  {s['prompt'][12:52]}"
    )

print("--- tone distance between movements (dB)")
m = len(shapes)
for i in range(m):
    print(" ".join(f"{float(np.abs(shapes[i] - shapes[j]).mean()):4.1f}" for j in range(m)))
print("--- harmony similarity between movements")
for i in range(m):
    print(" ".join(f"{float(chromas[i] @ chromas[j]):4.2f}" for j in range(m)))

print("--- windows")
records = [done[str(s["index"])] for s in segments]
attempts = [record["attempts"] for record in records]
retried = [s["index"] for s, n in zip(segments, attempts, strict=True) if n > candidates]
print("renders", sum(attempts), "of", candidates * len(segments), "planned; retried", retried)
print(
    "render seconds per window: mean", round(np.mean([r["render_s"] for r in records]), 1),
    "peak VRAM", max(r["peak_vram_gb"] for r in records),
    "worker RAM", max(r["worker_ram_gb"] for r in records),
)
for s, record in zip(segments, records, strict=True):
    a = record["assessment"]
    if a["timbre_jump_db"] > 3.5 or a["level_jump_db"] > 1.2 or a["dip_db"] > 2.0:
        print(
            f"  [{s['index']:2d}] {clock(s['start_s'])} {'NEW MOVEMENT ' if s['fresh'] else ''}"
            f"timbre {a['timbre_jump_db']:.1f} level {a['level_jump_db']:.1f} "
            f"dip {a['dip_db']:.1f} "
            f"busy {a['busyness']:.1f} trim {a['trim_db']:.1f}"
        )

print("--- short stops: 100 ms level more than 24 dB under the local level for 0.4 s or more")
with sf.SoundFile(wav) as f:
    sr = f.samplerate
    hop = sr // 10
    levels = []
    while True:
        x = f.read(hop, dtype="float32", always_2d=True)
        if not len(x):
            break
        levels.append(float(np.sqrt((x**2).mean())))
db = 20 * np.log10(np.array(levels) + 1e-9)
local = np.array([np.median(db[max(0, i - 80) : i + 80]) for i in range(len(db))])
under = db < local - 24
i = AFTER_START_S * 10
count = 0
longest = 0.0
while i < len(db) - BEFORE_END_S * 10:
    if under[i]:
        j = i
        while under[j + 1]:
            j += 1
        length = (j - i + 1) / 10
        longest = max(longest, length)
        if length >= 0.4:
            count += 1
            low = db[i : j + 1].min()
            print(f"  {clock(i / 10)}.{i % 10} {length:.1f} s, down to {low:.1f} dB")
        i = j + 1
    else:
        i += 1
print("stops", count, "longest", longest)
