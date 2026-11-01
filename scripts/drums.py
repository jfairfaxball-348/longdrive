"""How busy are the drums? Pictures of the average bar, and a few numbers to compare.

Onsets are folded into one bar of 16 slots, in the low end (30 to 150 Hz, kick and bass) and
on top (above 4 kHz, hats, percussion and the click of a kick). A plain four-on-the-floor over
offbeat hats looks like this, while a busy pattern fills in the gaps:

    low |@...@...@...@...|   top |@.+.@.+.@.+.@.+.|

- beat share: how much of the low end falls on the four beats, from 0.25 (even) to 1.
- top slots: how many of the 16 slots carry a strong hit on top.
- hits: sharp events per bar in each band. Bass notes count as low hits too.
- bar match: near 1 when every bar plays the same pattern, lower with fills and changes.
- mids: chords and keys, from 150 Hz to 4 kHz, against everything else. Further below zero
  means less music around the kick.

Give it a run folder to see every finished window of a piece.

    uv run python scripts/drums.py file.wav [bpm] [start_s] [stop_s]
    uv run python scripts/drums.py runs/<name>
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, stft

from longdrive import audio

HOP = 240  # 5 ms at 48 kHz
SLOTS = 16
MIN_GAP_S = 0.06
STRONG_SLOT = 0.5
# A hit has to rise this far compared with the strongest hits in the passage, in the log
# energy of its band, which leaves out the faint ticks that only a meter would notice.
HIT_SHARE = 0.35
SHADES = " .:-=+*#%@"


def onsets(mono, rate, low_hz, high_hz, size):
    freqs, _, spectrum = stft(
        mono, rate, nperseg=size, noverlap=size - HOP, boundary=None, padded=False
    )
    band = (freqs >= low_hz) & (freqs <= high_hz)
    energy = np.log10(np.square(np.abs(spectrum[band])).sum(axis=0) + 1e-9)
    return np.maximum(0.0, np.diff(energy, prepend=energy[0]))


def traces(x, rate):
    mono = x.mean(axis=1)
    return onsets(mono, rate, 30.0, 150.0, 2048), onsets(mono, rate, 4000.0, 16000.0, 512)


def hits_per_bar(trace, rate, bars):
    height = HIT_SHARE * np.percentile(trace, 99.8)
    return find_peaks(trace, height=height, distance=int(MIN_GAP_S * rate / HOP))[0].size / bars


def folded(trace, frames_per_bar):
    """The average bar in 16 slots, without its floor. A render can start anywhere in a bar."""
    positions = np.floor((np.arange(trace.size) % frames_per_bar) / frames_per_bar * SLOTS)
    pattern = np.bincount(positions.astype(int), trace, minlength=SLOTS)[:SLOTS].astype(float)
    return pattern - pattern.min()


def on_the_beat(pattern):
    """Turn a bar so that it starts on the strongest of the four possible sets of beats."""
    return np.roll(pattern, -int(np.argmax([pattern[p::4].sum() for p in range(4)])))


def bar_match(trace, frames_per_bar):
    """How close each bar's onsets are to the average bar."""
    bars = int(trace.size // frames_per_bar)
    if bars < 2:
        return float("nan")
    step = frames_per_bar / (2 * SLOTS)
    slots = np.floor(np.arange(trace.size) / step).astype(int)
    grid = np.bincount(slots, trace, minlength=bars * 2 * SLOTS)[: bars * 2 * SLOTS]
    grid = grid.reshape(bars, 2 * SLOTS)
    mean = grid.mean(axis=0)
    return float(np.mean([np.corrcoef(bar, mean)[0, 1] for bar in grid if bar.std() > 0]))


def mids_db(x, rate):
    levels = audio.band_levels(x, rate)
    edges = np.geomspace(40.0, 16_000.0, levels.size + 1)
    centres = np.sqrt(edges[:-1] * edges[1:])
    power = 10 ** (levels / 10)
    return float(10 * np.log10(power[(centres >= 150) & (centres < 4000)].sum() / power.sum()))


def drum_stats(x, rate, bpm):
    bar_s = 240.0 / bpm
    frames_per_bar = bar_s * rate / HOP
    bars = x.shape[0] / rate / bar_s
    low, top = traces(x, rate)
    low_bar = on_the_beat(folded(low, frames_per_bar))
    top_bar = folded(top, frames_per_bar)
    frames = min(low.size, top.size)
    return {
        "beat_share": float(low_bar[::4].sum() / max(low_bar.sum(), 1e-9)),
        "top_slots": int((top_bar > STRONG_SLOT * top_bar.max()).sum()),
        "low_hits": hits_per_bar(low, rate, bars),
        "top_hits": hits_per_bar(top, rate, bars),
        "bar_match": bar_match(low[:frames] + top[:frames], frames_per_bar),
        "busyness": audio.busyness(x, rate),
        "mids": mids_db(x, rate),
    }


def bar_picture(pattern):
    levels = pattern / max(pattern.max(), 1e-9)
    return "".join(SHADES[min(int(level * len(SHADES)), len(SHADES) - 1)] for level in levels)


def pictures(x, rate, bpm):
    """The low end and the top of the average bar, both turned to start on the beat."""
    frames_per_bar = 240.0 / bpm * rate / HOP
    low, top = traces(x, rate)
    low_bar, top_bar = folded(low, frames_per_bar), folded(top, frames_per_bar)
    turn = int(np.argmax([low_bar[p::4].sum() for p in range(4)]))
    return bar_picture(np.roll(low_bar, -turn)), bar_picture(np.roll(top_bar, -turn))


def line(stats):
    return (
        f"beat share {stats['beat_share']:4.2f}  top slots {stats['top_slots']:4.1f}  "
        f"hits low {stats['low_hits']:4.1f} top {stats['top_hits']:4.1f}  "
        f"bar match {stats['bar_match']:4.2f}  busyness {stats['busyness']:4.2f}  "
        f"mids {stats['mids']:5.1f}"
    )


def by_window(run):
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    for segment in plan["segments"]:
        path = run / "segments" / f"{segment['index']:05d}.wav"
        if not path.exists():
            break
        x, rate = audio.read_region(path, 0.0, segment["new_s"])
        low_picture, top_picture = pictures(x, rate, plan["bpm"])
        mark = "*" if segment["index"] == 0 or segment["fresh"] else " "
        minutes = segment["start_s"] / 60
        stats = line(drum_stats(x, rate, plan["bpm"]))
        print(f"{minutes:5.1f}{mark} |{low_picture}| |{top_picture}| {stats}")


if __name__ == "__main__":
    path = Path(sys.argv[1])
    if path.is_dir():
        by_window(path)
        sys.exit()
    bpm = float(sys.argv[2]) if len(sys.argv) > 2 else 110.0
    start = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    stop = float(sys.argv[4]) if len(sys.argv) > 4 else None
    x, rate = audio.read_region(path, start, stop)
    low_picture, top_picture = pictures(x, rate, bpm)
    print(f"low |{low_picture}|   top |{top_picture}|")
    print(line(drum_stats(x, rate, bpm)))
