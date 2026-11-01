"""A finer test than average tone: is the actual musical pattern changing?

Averages over a minute hide what a listener hears as change. Compare the note-by-note
pattern across a run of bars instead: how alike are any two passages, moment by moment?
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

path = Path(sys.argv[1])
bpm = int(sys.argv[2]) if len(sys.argv) > 2 else 110
x, sr = sf.read(path, dtype="float32", always_2d=True)
mono = x.mean(axis=1)
bar = int(60 / bpm * 4 * sr)
FFT = 16384
freqs = np.fft.rfftfreq(FFT, 1 / sr)
use = (freqs >= 110) & (freqs <= 1760)
pitch = (np.round(12 * np.log2(freqs[use] / 440.0)).astype(int) + 9) % 12
window = np.hanning(FFT)

step = bar // 4  # one beat
frames = []
for start in range(0, len(mono) - FFT, step):
    power = np.abs(np.fft.rfft(mono[start : start + FFT] * window))[use] ** 2
    c = np.zeros(12)
    np.add.at(c, pitch, power)
    frames.append(c / (np.linalg.norm(c) + 1e-9))
frames = np.array(frames)  # beats x 12

phrase = 32  # beats: eight bars
count = len(frames) // phrase
blocks = frames[: count * phrase].reshape(count, phrase, 12)
# flatten keeps the order of notes in time, so two passages only match if the pattern does
flat = blocks.reshape(count, -1)
flat = flat - flat.mean(axis=1, keepdims=True)
flat /= np.linalg.norm(flat, axis=1, keepdims=True) + 1e-9
sim = flat @ flat.T

print(f"{path.name}: {count} blocks of 8 bars")
every = max(1, count // 16)
to_first = [round(float(sim[0, i]), 2) for i in range(0, count, every)]
to_previous = [round(float(sim[i, i - 1]), 2) for i in range(1, count, every)]
print("similarity to the first passage:", to_first)
print("similarity to the previous one: ", to_previous)
near = np.mean([sim[i, i + 1] for i in range(count - 1)])
block_s = 8 * 4 * 60 / bpm
apart = [(i, j) for i in range(count) for j in range(count) if abs(i - j) * block_s > 180]
far = np.mean([sim[i, j] for i, j in apart])
print(f"mean similarity: neighbouring passages {near:.2f}, passages over 3 minutes apart {far:.2f}")
