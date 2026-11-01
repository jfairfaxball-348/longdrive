"""Join committed segments into the finished piece without holding it in memory."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

from longdrive import audio
from longdrive.plan import Plan

MP3_COMPRESSION = 0.1  # libsndfile scale: 0 is the highest bitrate
OUTPUT_FORMATS = ("wav", "flac", "mp3")

_BASS_HZ = 150.0
_BASS_SWAP_S = 1.0

# A plain WAV holds at most 4 GiB, about four hours of 24-bit stereo at 48 kHz.
_WAV_LIMIT_BYTES = 2**32 - 2**16
_BYTES_PER_FRAME = 6


@dataclass(frozen=True)
class MasterSettings:
    target_db: float = -16.0
    ride_s: float = 120.0
    max_ride_db: float = 9.0
    ceiling_db: float = -1.0
    fade_in_s: float = 8.0
    fade_out_s: float = 20.0


def crossfade(outgoing: np.ndarray, incoming: np.ndarray) -> np.ndarray:
    """Equal-power crossfade between the old window's tail and the new window's start.

    Each window tends to settle into its own texture, so even with tempo and harmony
    intact a hard join is a sudden change of tone. Both sides continue the same context
    in the same key and bar grid, so a slow fade turns that change into a gradual one.
    They diverge quickly, which is why they add as unrelated signals.
    """
    frames = min(outgoing.shape[0], incoming.shape[0])
    phase = (np.arange(frames, dtype=np.float32) + 0.5) / frames * (np.pi / 2)
    fade_out, fade_in = np.cos(phase)[:, None], np.sin(phase)[:, None]
    mixed = outgoing[:frames] * fade_out + incoming[:frames] * fade_in
    return np.concatenate([mixed, incoming[frames:]])


def mix_movements(outgoing: np.ndarray, incoming: np.ndarray, rate: int) -> np.ndarray:
    """Blend two different grooves the way a DJ does.

    Everything above the bass fades slowly across the whole blend. Two basslines at once
    would turn to mud, so the low end changes hands quickly at the midpoint instead.
    """
    frames = min(outgoing.shape[0], incoming.shape[0])
    outgoing, incoming = outgoing[:frames], incoming[:frames]
    lowpass = butter(4, _BASS_HZ, "low", fs=rate, output="sos")
    low_out = sosfiltfilt(lowpass, outgoing, axis=0).astype(np.float32)
    low_in = sosfiltfilt(lowpass, incoming, axis=0).astype(np.float32)

    time_s = (np.arange(frames, dtype=np.float32) - frames / 2) / rate
    swap = np.clip(time_s / _BASS_SWAP_S + 0.5, 0.0, 1.0)
    swap = (0.5 - 0.5 * np.cos(np.pi * swap))[:, None]
    return crossfade(outgoing - low_out, incoming - low_in) + low_out * (1 - swap) + low_in * swap


def iter_piece(
    plan: Plan, segment_paths: list[Path], block_frames: int = 48_000
) -> Iterator[np.ndarray]:
    """Yield the piece in order, crossfading each join over the previous tail."""
    if not segment_paths:
        return
    carry: np.ndarray | None = None
    for position, path in enumerate(segment_paths):
        segment = plan.segments[position]
        is_last = position == len(plan.segments) - 1
        with sf.SoundFile(path) as source:
            rate = source.samplerate
            body = int(round(segment.new_s * rate))

            if carry is not None:
                frames = min(carry.shape[0], body)
                head = audio.to_stereo(source.read(frames, dtype="float32", always_2d=True))
                yield mix_movements(carry, head, rate) if segment.fresh else crossfade(carry, head)
            remaining = body - (carry.shape[0] if carry is not None else 0)

            while remaining > 0:
                block = source.read(min(block_frames, remaining), dtype="float32", always_2d=True)
                if not block.shape[0]:
                    break
                remaining -= block.shape[0]
                yield audio.to_stereo(block)

            carry = None
            if not is_last:
                frames = int(round(segment.overlap_s * rate))
                carry = audio.to_stereo(source.read(frames, dtype="float32", always_2d=True))


def rechunk(blocks: Iterable[np.ndarray], frames: int) -> Iterator[np.ndarray]:
    """Regroup variable-sized blocks into blocks of exactly ``frames``."""
    pending: list[np.ndarray] = []
    held = 0
    for block in blocks:
        pending.append(block)
        held += block.shape[0]
        if held >= frames:
            joined = np.concatenate(pending)
            whole = (joined.shape[0] // frames) * frames
            for start in range(0, whole, frames):
                yield joined[start : start + frames]
            remainder = joined[whole:]
            pending, held = ([remainder], remainder.shape[0]) if remainder.size else ([], 0)
    if pending:
        yield np.concatenate(pending)


def level_ride(levels: np.ndarray, settings: MasterSettings) -> np.ndarray:
    """Per-second gain in dB: reach the target level and even out slow drift.

    The window is long on purpose. Swells lasting under a minute or so are the music
    breathing, and flattening them would make the piece feel lifeless.
    """
    power = np.square(levels)
    width = max(3, int(settings.ride_s) | 1)
    window = np.hanning(width + 2)[1:-1]
    window /= window.sum()
    padded = np.pad(power, width // 2, mode="edge")
    smoothed = np.convolve(padded, window, mode="valid")[: levels.size]
    gain_db = settings.target_db - 10.0 * np.log10(np.maximum(smoothed, 1e-10))
    # Clamp around the piece's overall offset rather than around zero, so a quiet
    # render can still reach the target while the ride itself stays bounded.
    centre = float(np.median(gain_db))
    return np.clip(gain_db, centre - settings.max_ride_db, centre + settings.max_ride_db)


def soft_limit(block: np.ndarray, ceiling: float) -> np.ndarray:
    """Transparent below the knee, then saturate smoothly so peaks never pass ceiling."""
    knee = 0.8 * ceiling
    span = ceiling - knee
    magnitude = np.abs(block)
    over = magnitude > knee
    if over.any():
        block = block.copy()
        block[over] = np.sign(block[over]) * (
            knee + span * np.tanh((magnitude[over] - knee) / span)
        )
    return block


def _fades(start: int, frames: int, total: int, rate: int, s: MasterSettings) -> np.ndarray:
    """Raised-cosine fade curve for ``frames`` samples beginning at ``start``."""
    position = start + np.arange(frames)
    curve = np.ones(frames, dtype=np.float32)
    if s.fade_in_s > 0:
        length = s.fade_in_s * rate
        curve *= 0.5 - 0.5 * np.cos(np.pi * np.clip(position / length, 0.0, 1.0))
    if s.fade_out_s > 0:
        length = s.fade_out_s * rate
        curve *= 0.5 - 0.5 * np.cos(np.pi * np.clip((total - position) / length, 0.0, 1.0))
    return curve


def render_master(
    plan: Plan,
    segment_paths: list[Path],
    out_stem: Path,
    settings: MasterSettings | None = None,
    formats: tuple[str, ...] = ("wav",),
) -> list[Path]:
    """Two streaming passes: measure the level curve, then write gained, limited audio."""
    settings = settings or MasterSettings()
    rate = sf.info(segment_paths[0]).samplerate

    levels, peaks, total = [], [], 0
    for block in rechunk(iter_piece(plan, segment_paths), rate):
        levels.append(audio.rms(block))
        peaks.append(float(np.abs(block).max()))
        total += block.shape[0]
    gain_db = level_ride(np.asarray(levels), settings)
    # Dynamic music has peaks far above its level, so lower the whole piece if reaching
    # the target would leave the limiter working on more than the occasional peak.
    peak_db = np.percentile(audio.db(np.asarray(peaks)) + gain_db, 95)
    gain_db = gain_db - max(0.0, float(peak_db) - settings.ceiling_db)
    block_centres = (np.arange(gain_db.size) + 0.5) * rate

    ceiling = 10 ** (settings.ceiling_db / 20)
    outputs, paths = _open_outputs(out_stem, rate, total, formats)
    try:
        written = 0
        for block in rechunk(iter_piece(plan, segment_paths), rate):
            frames = block.shape[0]
            position = written + np.arange(frames)
            gain = 10 ** (np.interp(position, block_centres, gain_db) / 20)
            curve = _fades(written, frames, total, rate, settings)
            shaped = block * (gain * curve)[:, None].astype(np.float32)
            finished = soft_limit(shaped, ceiling)
            for output in outputs:
                output.write(finished)
            written += frames
    finally:
        for output in outputs:
            output.close()
    return paths


def wav_container(frames: int) -> str:
    """Pick the file format for a WAV: RF64 is the extension of WAV past its 4 GiB limit."""
    return "RF64" if frames * _BYTES_PER_FRAME >= _WAV_LIMIT_BYTES else "WAV"


def _open_outputs(
    out_stem: Path, rate: int, frames: int, formats: tuple[str, ...]
) -> tuple[list[sf.SoundFile], list[Path]]:
    outputs, paths = [], []
    for name in formats:
        path = out_stem.with_suffix(f".{name}")
        if name == "wav":
            outputs.append(sf.SoundFile(path, "w", rate, 2, "PCM_24", format=wav_container(frames)))
        elif name == "flac":
            outputs.append(sf.SoundFile(path, "w", rate, 2, "PCM_24", format="FLAC"))
        elif name == "mp3":
            outputs.append(
                sf.SoundFile(
                    path,
                    "w",
                    rate,
                    2,
                    "MPEG_LAYER_III",
                    format="MP3",
                    compression_level=MP3_COMPRESSION,
                )
            )
        else:
            raise ValueError(f"unknown output format: {name}")
        paths.append(path)
    return outputs, paths
