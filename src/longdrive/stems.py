"""Swap the model's drums for a plain, quiet kick, so that they stop crowding the music.

Words barely reach the drums, and steering the model with a beat only lasts one window.
So the finished piece is split with Demucs, and its drum stem is subtracted from the
original rather than rebuilt from stems, which leaves everything else as it was rendered.
The stem is kept in the run, so that trying another level only needs a new master.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from longdrive import audio
from longdrive.engines import EngineError
from longdrive.engines.ace_step import ENGINE_DIR, engine_python
from longdrive.master import iter_piece
from longdrive.plan import Plan

WORKER = Path(__file__).parent / "engines" / "stems_worker.py"
SPLIT_TIMEOUT_S = 6 * 60 * 60

_KICK_S = 0.2
_MIN_DRUMS_DB = -40.0

log = logging.getLogger("longdrive")


@dataclass(frozen=True)
class DrumSwap:
    """Turn a piece's drums down as it streams past, and lay a kick in their place."""

    drums_path: Path
    drums_db: float
    kick: np.ndarray
    kick_frames: np.ndarray

    def apply(self, blocks: Iterable[np.ndarray]) -> Iterator[np.ndarray]:
        cut = 1.0 - 10 ** (self.drums_db / 20)
        position = 0
        with sf.SoundFile(self.drums_path) as drums:
            for block in blocks:
                frames = block.shape[0]
                stem = audio.to_stereo(drums.read(frames, dtype="float32", always_2d=True))
                hits = kick_track(self.kick, self.kick_frames, position, frames)
                yield block - stem * cut + hits[:, None]
                position += frames


def soft_kick(rate: int) -> np.ndarray:
    """A short round kick, with no click on top to take attention from the music."""
    length = int(_KICK_S * rate)
    time_s = np.arange(length) / rate
    sweep = 52.0 + 60.0 * np.exp(-time_s / 0.02)
    attack = np.minimum(1.0, time_s / 0.002)
    release = np.minimum(1.0, (length - np.arange(length)) / (0.01 * rate))
    return (np.sin(2 * np.pi * np.cumsum(sweep) / rate) * np.exp(-time_s / 0.07)) * attack * release


def kick_track(kick: np.ndarray, kick_frames: np.ndarray, start: int, frames: int) -> np.ndarray:
    """The part of the kick track that falls between ``start`` and ``start + frames``."""
    track = np.zeros(frames, dtype=np.float32)
    first = np.searchsorted(kick_frames, start - kick.size)
    last = np.searchsorted(kick_frames, start + frames)
    for hit in kick_frames[first:last] - start:
        begin, stop = max(0, hit), min(frames, hit + kick.size)
        track[begin:stop] += kick[begin - hit : stop - hit]
    return track


def kick_times(plan: Plan, drums_path: Path, duration_s: float) -> np.ndarray:
    """A kick on every beat, easing from each section's grid to the next through the blend.

    A jump between grids would be a stumble in the pulse. Spread across a blend, even half a
    beat is a change of tempo too small to hear. A section with no drums has no grid of its
    own, so it carries on with its neighbour's.
    """
    beat_s = 60.0 / plan.bpm
    blend_s = plan.timing.blend_s
    rate = sf.info(drums_path).samplerate
    starts = [s.start_s for s in plan.segments if s.index == 0 or s.fresh]
    starts = [start for start in starts if start < duration_s]
    ends = [*starts[1:], duration_s]

    grids: list[float | None] = []
    for number, (start, end) in enumerate(zip(starts, ends, strict=True)):
        # Measure past the blend into the section, where nothing else is playing.
        measured_from = start + blend_s if number and start + blend_s < end else start
        passage, _ = audio.read_region(drums_path, measured_from, end)
        if audio.db(audio.rms(passage)) < _MIN_DRUMS_DB:
            grids.append(None)
        else:
            grids.append(measured_from + audio.beat_phase(passage, rate, beat_s))

    known = [grid for grid in grids if grid is not None]
    phases = [known[0] % beat_s if known else 0.0]
    for grid in grids[1:]:
        if grid is None:
            phases.append(phases[-1])
        else:
            phases.append(grid + round((phases[-1] - grid) / beat_s) * beat_s)

    knots_s = [time for start in starts[1:] for time in (start, start + blend_s)]
    knots = [phase for pair in zip(phases, phases[1:], strict=False) for phase in pair]
    beats = np.arange(0.0, duration_s + 2 * beat_s, beat_s)
    times = beats + (np.interp(beats, knots_s, knots) if knots else phases[0])
    return times[(times >= 0) & (times < duration_s)]


def split_drums(piece_path: Path, drums_path: Path) -> None:
    environment = {**os.environ, "HF_HUB_OFFLINE": "1"}
    command = [str(engine_python()), str(WORKER), str(piece_path), str(drums_path)]
    try:
        subprocess.run(
            command, cwd=ENGINE_DIR, env=environment, check=True, timeout=SPLIT_TIMEOUT_S,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
    except subprocess.CalledProcessError as error:
        raise EngineError(f"splitting the drums failed: {error.stderr[-500:]}") from error


def _write_piece(plan: Plan, segment_paths: list[Path], path: Path, rate: int) -> int:
    frames = 0
    with sf.SoundFile(path, "w", rate, 2, "FLOAT", format="RF64") as target:
        for block in iter_piece(plan, segment_paths):
            target.write(block)
            frames += block.shape[0]
    return frames


def _music_level(plan: Plan, segment_paths: list[Path], drums_path: Path) -> float:
    power, frames = 0.0, 0
    with sf.SoundFile(drums_path) as drums:
        for block in iter_piece(plan, segment_paths):
            stem = audio.to_stereo(drums.read(block.shape[0], dtype="float32", always_2d=True))
            rest = block[: stem.shape[0]] - stem
            power += float(np.square(rest).mean(axis=1).sum())
            frames += rest.shape[0]
    return float(np.sqrt(power / max(1, frames)))


def prepare(plan: Plan, segment_paths: list[Path], work_dir: Path) -> DrumSwap:
    """Split the piece if that hasn't been done yet, and work out where the kicks go."""
    rate = sf.info(segment_paths[0]).samplerate
    expected = sum(round(s.new_s * rate) for s in plan.segments[: len(segment_paths)])
    drums_path = work_dir / "drums.wav"
    if not drums_path.exists() or abs(sf.info(drums_path).frames - expected) > rate:
        log.info("Splitting out the drums")
        piece_path = work_dir / "piece.wav"
        _write_piece(plan, segment_paths, piece_path, rate)
        split_drums(piece_path, drums_path)
        piece_path.unlink(missing_ok=True)

    duration_s = sf.info(drums_path).frames / rate
    level = _music_level(plan, segment_paths, drums_path)
    kick = soft_kick(rate)
    if plan.kick_db is None:
        return DrumSwap(drums_path, _drums_db(plan), kick, np.zeros(0, dtype=np.int64))
    # Level the kick as a whole track against the music, gaps included.
    track_level = np.sqrt(np.square(kick).sum() / (60.0 / plan.bpm * rate))
    kick = (kick * level / track_level * 10 ** (plan.kick_db / 20)).astype(np.float32)
    times = kick_times(plan, drums_path, duration_s)
    return DrumSwap(drums_path, _drums_db(plan), kick, np.round(times * rate).astype(np.int64))


def _drums_db(plan: Plan) -> float:
    return 0.0 if plan.drums_db is None else plan.drums_db
