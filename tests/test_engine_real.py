"""The ACE-Step worker against the real model. Run with: uv run pytest -m engine"""

import numpy as np
import pytest

from longdrive import audio
from longdrive.engines import RenderRequest, Settings
from longdrive.engines.ace_step import AceStepEngine, missing_parts

pytestmark = pytest.mark.engine

CAPTION = "soft piano and warm pads, instrumental, slow tempo"
RATE = 48_000
SPLICE_S = 0.1


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    settings = Settings()
    if missing_parts(settings):
        pytest.skip("ACE-Step is not installed")
    instance = AceStepEngine(settings, log_path=tmp_path_factory.mktemp("log") / "engine.log")
    instance.start()
    yield instance
    instance.stop()


def request(path, **overrides) -> RenderRequest:
    values = dict(
        out_path=path,
        caption=CAPTION,
        seed=7,
        new_s=24.0,
        bpm=60,
        key="D major",
        time_signature="4",
    )
    return RenderRequest(**{**values, **overrides})


def test_render_from_nothing(engine, tmp_path):
    path = tmp_path / "first.wav"
    reply = engine.render(request(path), timeout_s=900)
    data, rate = audio.read_region(path)
    assert rate == RATE
    assert data.shape[0] == pytest.approx(24 * RATE, abs=RATE // 10)
    # The model fades out as a window ends, so only the first part is judged.
    assert audio.assess(data[: 12 * rate], rate).ok
    assert reply["peak_vram_gb"] < 4.0


def test_continuation_returns_the_context_untouched(engine, tmp_path):
    first = tmp_path / "first.wav"
    engine.render(request(first), timeout_s=900)
    data, rate = audio.read_region(first)

    context_path = tmp_path / "context.wav"
    context = data[-8 * rate :]
    audio.write_audio(context_path, context, rate)

    second = tmp_path / "second.wav"
    engine.render(
        request(second, seed=8, context_path=context_path, context_s=8.0), timeout_s=900
    )
    output, _ = audio.read_region(second)
    assert output.shape[0] == pytest.approx(32 * RATE, abs=RATE // 10)

    # The last few milliseconds of the context are blended into the new audio, and
    # upstream clamps its input to full scale. That audio is never kept, so neither matters.
    kept = (8 * rate) - int(SPLICE_S * rate)
    assert np.allclose(output[:kept], np.clip(context[:kept], -1, 1), atol=1e-4)
    assert audio.assess(output[8 * rate : 20 * rate], rate, context).ok


def test_same_seed_repeats_exactly(engine, tmp_path):
    first, second = tmp_path / "a.wav", tmp_path / "b.wav"
    engine.render(request(first), timeout_s=900)
    engine.render(request(second), timeout_s=900)
    a, _ = audio.read_region(first)
    b, _ = audio.read_region(second)
    assert np.allclose(a, b, atol=1e-4)
