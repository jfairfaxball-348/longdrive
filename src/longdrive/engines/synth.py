"""A dependency-free engine: slow additive drones, no model and no GPU.

It exists so the whole pipeline can be exercised in seconds. Phases come from the
absolute position in the piece, so consecutive renders join seamlessly the way a
well-behaved model continuation should.
"""

from __future__ import annotations

import numpy as np

from longdrive import audio
from longdrive.engines.base import RenderRequest

SAMPLE_RATE = 48_000

_RATIOS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)


class SynthEngine:
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def render(self, request: RenderRequest, timeout_s: float) -> dict:
        generated = self.generate(request)
        if request.context_path is None:
            output = generated
        else:
            context, _ = audio.read_region(request.context_path)
            output = np.concatenate([context, generated])
        audio.write_audio(request.out_path, output, SAMPLE_RATE)
        return {}

    def generate(self, request: RenderRequest) -> np.ndarray:
        frames = int(round(request.new_s * SAMPLE_RATE))
        t = request.piece_start_s + np.arange(frames) / SAMPLE_RATE
        root_hz = 55.0 * 2 ** ((sum(request.key.encode()) % 12) / 12)

        # Tying the voices to the plan's key rather than the render seed keeps the
        # drone the same across windows, as a real continuation would be.
        rng = np.random.default_rng(sum(request.key.encode()))
        left = np.zeros(frames, dtype=np.float64)
        right = np.zeros(frames, dtype=np.float64)
        for ratio in _RATIOS:
            rate_hz = rng.uniform(0.01, 0.05)
            phase = rng.uniform(0, 2 * np.pi, 2)
            swell = 0.5 + 0.5 * np.sin(2 * np.pi * rate_hz * t + phase[0])
            tone = np.sin(2 * np.pi * root_hz * ratio * t + phase[1]) * swell / ratio
            pan = rng.uniform(0.3, 0.7)
            left += tone * (1 - pan)
            right += tone * pan
        return (0.2 * np.stack([left, right], axis=1)).astype(np.float32)
