from __future__ import annotations

import numpy as np
import pytest

from longdrive.engines import RenderRequest
from longdrive.engines.synth import SAMPLE_RATE, SynthEngine
from longdrive.plan import Timing, make_plan
from longdrive.run import Run

# Short windows keep the suite fast while exercising the same joins as a real run.
FAST_TIMING = Timing(hop_s=32.0, context_s=32.0, tail_s=40.0, crossfade_s=4.0, blend_s=32.0)


@pytest.fixture
def rate() -> int:
    return SAMPLE_RATE


@pytest.fixture
def tone(rate):
    def make(seconds: float, level: float = 0.2, hz: float = 220.0) -> np.ndarray:
        t = np.arange(int(seconds * rate)) / rate
        wave = (level * np.sin(2 * np.pi * hz * t)).astype(np.float32)
        return np.stack([wave, wave], axis=1)

    return make


@pytest.fixture
def new_run(tmp_path):
    def make(duration_s: float = 80.0, candidates: int = 1, name: str = "piece") -> Run:
        plan = make_plan("slow test drone", duration_s, seed=11, bpm=60, timing=FAST_TIMING)
        from longdrive.engines import Settings

        return Run.create(tmp_path / name, plan, "synth", Settings(), candidates)

    return make


class ScriptedEngine(SynthEngine):
    """A synth engine that can fail on chosen calls and records what it was asked."""

    def __init__(self, failures: dict[int, Exception] | None = None) -> None:
        self.failures = failures or {}
        self.calls = 0
        self.requests: list[RenderRequest] = []
        self.starts = 0
        self.stops = 0

    def start(self) -> None:
        self.starts += 1

    def stop(self) -> None:
        self.stops += 1

    def render(self, request: RenderRequest, timeout_s: float) -> None:
        self.calls += 1
        self.requests.append(request)
        failure = self.failures.get(self.calls)
        if failure:
            raise failure
        super().render(request, timeout_s)


@pytest.fixture
def scripted():
    return ScriptedEngine
