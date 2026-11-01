"""Audio engines: the models that turn one render request into audio."""

from __future__ import annotations

from pathlib import Path

from longdrive.engines.base import (
    Engine,
    EngineError,
    EngineFatalError,
    RenderRequest,
    Settings,
)


def create_engine(name: str, settings: Settings, log_path: Path | None = None) -> Engine:
    """Build an engine by name. Imports are local so the core stays free of model code."""
    if name == "ace-step":
        from longdrive.engines.ace_step import AceStepEngine

        return AceStepEngine(settings, log_path=log_path)
    if name == "synth":
        from longdrive.engines.synth import SynthEngine

        return SynthEngine()
    raise ValueError(f"unknown engine: {name}")


__all__ = [
    "Engine",
    "EngineError",
    "EngineFatalError",
    "RenderRequest",
    "Settings",
    "create_engine",
]
