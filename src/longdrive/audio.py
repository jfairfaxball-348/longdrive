"""Audio file helpers and the signal checks used to pick between candidate renders."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft

# 32-bit float keeps peaks above full scale intact. Normalising each render would
# shift the level of the context region and break the join to the audio before it.
INTERMEDIATE_SUBTYPE = "FLOAT"

DEAD_AIR_DB = -60.0
MAX_DEAD_AIR_S = 4.0
MAX_LEVEL_JUMP_DB = 12.0
MAX_DRIFT_DB = 12.0
MAX_TRIM_DB = 12.0
MAX_DIP_DB = 9.0

# Scoring weights. A movement should hold its energy: no busier or emptier than it
# started, no breakdowns, and no sudden change of tone at a join. Raw loudness counts a
# little, since a louder render often has more going on.
_RISE_WEIGHT = 0.5
_BUSYNESS_WEIGHT = 3.0
_DIP_WEIGHT = 1.0
_JOIN_WEIGHT = 1.0

_DIP_BLOCK_S = 8.0

_BEAT_HOP = 240  # 5 ms at 48 kHz
_ONSET_FFT = 2048
_KICK_BAND_HZ = (30.0, 150.0)

_BAND_COUNT = 24
_BAND_LOW_HZ = 40.0
_BAND_HIGH_HZ = 16_000.0
_FFT_SIZE = 4096


def read_region(
    path: Path, start_s: float = 0.0, stop_s: float | None = None
) -> tuple[np.ndarray, int]:
    """Read ``[start_s, stop_s)`` as float32 stereo frames."""
    with sf.SoundFile(path) as source:
        rate = source.samplerate
        start = int(round(start_s * rate))
        stop = source.frames if stop_s is None else int(round(stop_s * rate))
        source.seek(min(start, source.frames))
        audio = source.read(max(0, stop - start), dtype="float32", always_2d=True)
    return to_stereo(audio), rate


def write_audio(path: Path, audio: np.ndarray, rate: int) -> None:
    """Write an intermediate file; the rename means readers never see a partial one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.partial")
    sf.write(temporary, audio, rate, format="WAV", subtype=INTERMEDIATE_SUBTYPE)
    os.replace(temporary, path)


def to_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1)
    return audio[:, :2]


def db(value: float | np.ndarray) -> float | np.ndarray:
    return 20.0 * np.log10(np.maximum(value, 1e-10))


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0


def block_rms(audio: np.ndarray, rate: int, block_s: float = 1.0) -> np.ndarray:
    """RMS of consecutive blocks, ignoring a final partial block."""
    size = max(1, int(rate * block_s))
    count = audio.shape[0] // size
    if count == 0:
        return np.array([rms(audio)])
    power = np.square(audio[: count * size]).mean(axis=1).reshape(count, size).mean(axis=1)
    return np.sqrt(power)


def _band_power(audio: np.ndarray, rate: int) -> np.ndarray:
    """Power in log-spaced bands for each analysis frame, shaped (frames, bands)."""
    mono = audio.mean(axis=1)
    if mono.size < _FFT_SIZE:
        mono = np.pad(mono, (0, _FFT_SIZE - mono.size))
    starts = np.arange(0, mono.size - _FFT_SIZE + 1, _FFT_SIZE // 2)
    window = np.hanning(_FFT_SIZE)
    freqs = np.fft.rfftfreq(_FFT_SIZE, 1.0 / rate)
    edges = np.geomspace(_BAND_LOW_HZ, min(_BAND_HIGH_HZ, rate / 2), _BAND_COUNT + 1)
    bins = np.searchsorted(freqs, edges)

    bands = np.zeros((starts.size, _BAND_COUNT))
    for index, start in enumerate(starts):
        power = np.square(np.abs(np.fft.rfft(mono[start : start + _FFT_SIZE] * window)))
        for band in range(_BAND_COUNT):
            low, high = bins[band], max(bins[band + 1], bins[band] + 1)
            bands[index, band] = power[low:high].mean()
    return bands


def band_levels(audio: np.ndarray, rate: int) -> np.ndarray:
    """Mean level in dB across log-spaced bands: a coarse picture of the timbre."""
    return 10.0 * np.log10(np.maximum(_band_power(audio, rate).mean(axis=0), 1e-12))


def _kick_onsets(audio: np.ndarray, rate: int) -> np.ndarray:
    """Rises in low-end energy, at 5 ms resolution: a trace of where the kicks fall."""
    mono = audio.mean(axis=1)
    freqs, _, spectrum = stft(
        mono, rate, nperseg=_ONSET_FFT, noverlap=_ONSET_FFT - _BEAT_HOP, boundary=None, padded=False
    )
    band = (freqs >= _KICK_BAND_HZ[0]) & (freqs <= _KICK_BAND_HZ[1])
    energy = np.log10(np.square(np.abs(spectrum[band])).sum(axis=0) + 1e-9)
    onsets = np.maximum(0.0, np.diff(energy, prepend=energy[0]))
    return onsets - onsets.mean()


def beat_offset(outgoing: np.ndarray, incoming: np.ndarray, rate: int, bar_s: float) -> int:
    """Frames to drop from ``incoming`` so that its beats land on those of ``outgoing``.

    Separate renders hold the same tempo to within about 0.03 BPM but start at an
    arbitrary point in the bar, so a single offset of up to one bar lines them up.
    """
    reference = _kick_onsets(outgoing, rate)
    candidate = _kick_onsets(incoming, rate)
    steps = int(bar_s * rate / _BEAT_HOP)
    span = min(reference.size, candidate.size - steps)
    if span <= 0:
        return 0
    scores = [float(reference[:span] @ candidate[lag : lag + span]) for lag in range(steps)]
    return int(np.argmax(scores)) * _BEAT_HOP


def beat_phase(audio: np.ndarray, rate: int, beat_s: float) -> float:
    """Where the beats fall in a passage, in seconds from its start, from its low-end onsets."""
    onsets = np.maximum(0.0, _kick_onsets(audio, rate))
    frames_per_beat = beat_s * rate / _BEAT_HOP
    positions = np.floor(np.arange(onsets.size) % frames_per_beat).astype(int)
    folded = np.bincount(positions, onsets, minlength=int(frames_per_beat) + 1)
    # A beat is several frames wide, so judge each position with its neighbours.
    wrapped = np.concatenate([folded[-2:], folded, folded[:2]])
    peak = int(np.argmax(np.convolve(wrapped, np.ones(5), "valid")))
    return (peak * _BEAT_HOP + _ONSET_FFT // 2) / rate % beat_s


def spectral_shape(audio: np.ndarray, rate: int) -> np.ndarray:
    """Band levels with the overall loudness removed, leaving only the tone colour."""
    levels = band_levels(audio, rate)
    return levels - levels.mean()


def busyness(audio: np.ndarray, rate: int) -> float:
    """Average rise in band energy between frames, in dB: higher means more is happening.

    Sustained sound stays near zero. Notes, drums and other onsets push it up, which
    makes it a usable stand-in for how far a passage is from "slow moving".
    """
    levels = 10.0 * np.log10(np.maximum(_band_power(audio, rate), 1e-12))
    return float(np.maximum(0.0, np.diff(levels, axis=0)).mean())


def level_dip(audio: np.ndarray, rate: int) -> float:
    """How far the quietest few seconds fall below the rest: a breakdown shows up here."""
    levels = db(block_rms(audio, rate, _DIP_BLOCK_S))
    return float(np.median(levels) - levels.min())


def longest_quiet_run_s(audio: np.ndarray, rate: int, threshold_db: float = DEAD_AIR_DB) -> float:
    """Length of the longest stretch below ``threshold_db``, at 0.25 s resolution."""
    block_s = 0.25
    quiet = db(block_rms(audio, rate, block_s)) < threshold_db
    longest = run = 0
    for is_quiet in quiet:
        run = run + 1 if is_quiet else 0
        longest = max(longest, run)
    return longest * block_s


@dataclass(frozen=True)
class Character:
    """How a movement starts: what everything after it in that movement holds on to."""

    shape: tuple[float, ...]
    level_db: float
    busyness: float

    @classmethod
    def measure(cls, audio: np.ndarray, rate: int) -> Character:
        shape = tuple(round(float(band), 3) for band in spectral_shape(audio, rate))
        return cls(shape, float(db(rms(audio))), busyness(audio, rate))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Character:
        return cls(tuple(data["shape"]), data["level_db"], data["busyness"])


@dataclass
class Assessment:
    """Signal-level checks on one candidate. A lower ``score`` is a better fit."""

    score: float
    trim_db: float
    level_db: float
    busyness: float
    drift_db: float
    dip_db: float
    level_jump_db: float
    timbre_jump_db: float
    dead_air_s: float
    peak: float
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict:
        return asdict(self)


def trim_to(level_db: float, target_db: float) -> float:
    """The gain that takes a level to a target, within the limits of what is sensible."""
    return float(np.clip(target_db - level_db, -MAX_TRIM_DB, MAX_TRIM_DB))


def assess(
    new_audio: np.ndarray,
    rate: int,
    previous_audio: np.ndarray | None = None,
    character: Character | None = None,
    window_s: float = 8.0,
) -> Assessment:
    """Score ``new_audio`` by how well it continues the piece.

    Left alone, the model treats any piece as a song in progress: calm music gains drums
    and bass within minutes, and a steady groove falls into breakdowns. Comparing every
    candidate with ``character`` and keeping the closest is a restoring force, which
    holds a movement at its starting character. Change comes from new movements.

    Level is not a reason to reject a render. It varies by several dB between seeds and
    does not depend on the context, so ``trim_db`` is the gain that brings a candidate
    to the level of the piece, and the join is judged as if that gain were already applied.

    These checks guard against failures and drift, but know nothing about whether the
    result sounds good. The join window spans a couple of bars at slow tempos: joins
    fall on bar lines, so a short window would compare a decaying note with the next
    chord and report a jump at every ordinary downbeat.
    """
    problems: list[str] = []
    if not np.isfinite(new_audio).all():
        problems.append("non-finite samples")
        new_audio = np.nan_to_num(new_audio)

    peak = float(np.abs(new_audio).max()) if new_audio.size else 0.0
    level = float(db(rms(new_audio)))
    calm = busyness(new_audio, rate)
    dip = level_dip(new_audio, rate)
    dead_air_s = longest_quiet_run_s(new_audio, rate)
    if dead_air_s > MAX_DEAD_AIR_S:
        problems.append(f"{dead_air_s:.1f}s of dead air")
    if dip > MAX_DIP_DB:
        problems.append(f"{dip:.1f} dB breakdown")

    trim = 0.0
    if character is not None:
        trim = trim_to(level, character.level_db)

    level_jump = timbre_jump = 0.0
    if previous_audio is not None and previous_audio.size:
        span = int(window_s * rate)
        before, after = previous_audio[-span:], new_audio[:span] * 10 ** (trim / 20)
        level_jump = abs(float(db(rms(after))) - float(db(rms(before))))
        timbre_jump = float(np.abs(band_levels(after, rate) - band_levels(before, rate)).mean())
        if level_jump > MAX_LEVEL_JUMP_DB:
            problems.append(f"{level_jump:.1f} dB level jump")

    if character is None:
        drift = 0.0
        score = _BUSYNESS_WEIGHT * calm + _DIP_WEIGHT * dip + dead_air_s
    else:
        drift = float(np.abs(spectral_shape(new_audio, rate) - np.asarray(character.shape)).mean())
        if drift > MAX_DRIFT_DB:
            problems.append(f"{drift:.1f} dB drift from the piece so far")
        score = (
            drift
            + _BUSYNESS_WEIGHT * abs(calm - character.busyness)
            + _RISE_WEIGHT * max(0.0, level - character.level_db)
            + _DIP_WEIGHT * dip
            + _JOIN_WEIGHT * (level_jump + timbre_jump)
            + dead_air_s
        )
    return Assessment(
        score, trim, level, calm, drift, dip, level_jump, timbre_jump, dead_air_s, peak, problems
    )
