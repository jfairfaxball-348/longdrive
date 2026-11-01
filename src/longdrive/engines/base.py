"""The contract between the conductor and an audio engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


class EngineError(RuntimeError):
    """A render failed in a way that a restart or a new seed may fix."""


class EngineFatalError(RuntimeError):
    """The engine cannot work at all, so retrying would only waste the night."""


@dataclass(frozen=True)
class Settings:
    """Model choice and sampler settings, fixed for a whole run."""

    model: str = "acestep-v15-turbo"
    steps: int = 8
    guidance: float = 1.0
    shift: float = 3.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RenderRequest:
    """One window of audio.

    The engine writes ``context_s + new_s`` seconds to ``out_path``: the context
    unchanged, followed by ``new_s`` of generated audio. With no ``context_path``
    the output is ``new_s`` long and starts from nothing.
    """

    out_path: Path
    caption: str
    seed: int
    new_s: float
    bpm: int
    key: str
    time_signature: str
    piece_start_s: float = 0.0
    context_path: Path | None = None
    context_s: float = 0.0

    def to_payload(self) -> dict:
        payload = asdict(self)
        for name in ("out_path", "context_path"):
            value = payload[name]
            payload[name] = None if value is None else str(value)
        return payload


class Engine(Protocol):
    def start(self) -> None:
        """Load the model. Must be safe to call again after ``stop``."""

    def render(self, request: RenderRequest, timeout_s: float) -> dict:
        """Write the requested audio and return diagnostics, or raise ``EngineError``."""

    def stop(self) -> None:
        """Release the model. Must be safe to call at any time."""
