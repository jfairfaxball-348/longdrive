"""At each blend, are the kicks of the two movements really on top of each other?"""
import json
import sys
from pathlib import Path

import numpy as np

from longdrive import audio

run = Path(sys.argv[1])
plan = json.loads((run / "plan.json").read_text())
bar = 60 / plan["bpm"] * 4
beat = bar / 4
segments = plan["segments"]
for segment in segments:
    if not segment["fresh"]:
        continue
    previous = segments[segment["index"] - 1]
    a, sr = audio.read_region(
        run / "segments" / f"{previous['index']:05d}.wav",
        previous["new_s"],
        previous["new_s"] + previous["overlap_s"],
    )
    b, _ = audio.read_region(
        run / "segments" / f"{segment['index']:05d}.wav", 0, previous["overlap_s"]
    )
    ea, eb = audio._kick_onsets(a, sr), audio._kick_onsets(b, sr)
    n = min(ea.size, eb.size)
    print(f"blend into movement at {segment['start_s']/60:.1f} min ({segment['key']})")
    # In sections through the blend: tight all the way through, or drifting apart?
    third = n // 3
    for name, s in (("start", 0), ("middle", third), ("end", 2 * third)):
        x, y = ea[s : s + third], eb[s : s + third]
        c = np.correlate(y - y.mean(), x - x.mean(), "full")
        mid = x.size - 1
        steps = int(beat * sr / 240 / 2)
        window = c[mid - steps : mid + steps + 1]
        lag = (int(np.argmax(window)) - steps) * 5
        zero = c[mid] / (np.sqrt((x**2).sum() * (y**2).sum()) + 1e-9)
        print(f"  {name:6s} best lag {lag:5d} ms, onset match at zero lag {zero:5.2f}")
