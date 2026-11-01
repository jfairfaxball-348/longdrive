"""How much does a piece actually change over time? Tone, harmony and density, minute by minute."""
import sys

import numpy as np
import soundfile as sf

from longdrive import audio

NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
FFT = 8192


def chroma(mono, sr):
    """Energy per pitch class between 80 Hz and 2 kHz."""
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
    best, name = -2.0, ""
    for root in range(12):
        for profile, mode in ((MAJOR, "maj"), (MINOR, "min")):
            r = np.corrcoef(c, np.roll(profile, root))[0, 1]
            if r > best:
                best, name = r, f"{NAMES[root]} {mode}"
    return name


path = sys.argv[1]
block_s = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
x, sr = sf.read(path, dtype="float32", always_2d=True)
b = int(block_s * sr)
n = len(x) // b
shapes, chromas, busy = [], [], []
for i in range(n):
    seg = x[i * b : (i + 1) * b]
    shapes.append(audio.spectral_shape(seg, sr))
    chromas.append(chroma(seg.mean(axis=1), sr))
    busy.append(round(audio.busyness(seg, sr), 1))
print(f"{len(x)/sr/60:.1f} min in {block_s:.0f}s blocks")
print("tone vs start ", [round(float(np.abs(s - shapes[0]).mean()), 1) for s in shapes])
steps = [float(np.abs(shapes[i] - shapes[i - 1]).mean()) for i in range(1, n)]
print("tone step     ", [round(s, 1) for s in steps])
print("harmony vs start", [round(float(chromas[0] @ c), 2) for c in chromas])
print("key estimate  ", [key_of(c) for c in chromas])
print("busyness      ", busy)
