import numpy as np
import pytest
import soundfile as sf

from longdrive import audio
from longdrive.engines import RenderRequest
from longdrive.engines.synth import SynthEngine
from longdrive.master import (
    MasterSettings,
    crossfade,
    iter_piece,
    level_ride,
    mix_movements,
    rechunk,
    render_master,
    soft_limit,
    wav_container,
)
from longdrive.plan import Timing, make_plan, parse_journey


def write_segments(tmp_path, plan, rate):
    """Render a plan with the synth, stored the way the conductor commits segments."""
    engine = SynthEngine()
    paths = []
    for segment in plan.segments:
        path = tmp_path / f"{segment.index}.wav"
        request = RenderRequest(
            out_path=path,
            caption="",
            seed=0,
            new_s=segment.new_s + plan.timing.tail_s,
            bpm=60,
            key=plan.key,
            time_signature="4",
            piece_start_s=segment.start_s,
        )
        audio.write_audio(path, engine.generate(request), rate)
        paths.append(path)
    return paths


@pytest.fixture
def plan():
    timing = Timing(hop_s=32, context_s=32, tail_s=40, crossfade_s=4, blend_s=32)
    return make_plan("drone", 80, seed=3, bpm=60, key="D minor", timing=timing)


def test_crossfade_starts_on_the_old_audio_and_ends_on_the_new(rate):
    outgoing = np.full((rate, 2), 0.5, dtype=np.float32)
    incoming = np.full((rate, 2), -0.5, dtype=np.float32)
    mixed = crossfade(outgoing, incoming)
    assert mixed[0, 0] == pytest.approx(0.5, abs=1e-3)
    assert mixed[-1, 0] == pytest.approx(-0.5, abs=1e-3)


def test_a_blend_starts_on_the_old_movement_and_ends_on_the_new(tone, rate):
    old, new = tone(32, hz=400), tone(32, hz=900)
    mixed = mix_movements(old, new, rate)
    assert np.abs(mixed[:2048] - old[:2048]).max() < 0.02
    assert np.abs(mixed[-2048:] - new[-2048:]).max() < 0.02


def test_a_blend_never_plays_two_basslines_at_once(tone, rate):
    old, new = tone(32, level=0.4, hz=60), tone(32, level=0.4, hz=90)
    mixed = mix_movements(old, new, rate)
    quarter = 8 * rate
    assert audio.rms(mixed[quarter:quarter + rate]) == pytest.approx(audio.rms(old), rel=0.05)
    three_quarters = 24 * rate
    assert audio.rms(mixed[three_quarters:three_quarters + rate]) == pytest.approx(
        audio.rms(new), rel=0.05
    )


def test_a_blend_fades_everything_above_the_bass_slowly(tone, rate):
    old, new = tone(32, level=0.4, hz=1000), tone(32, level=0.4, hz=1500)
    mixed = mix_movements(old, new, rate)
    middle = 16 * rate
    assert audio.rms(mixed[middle:middle + rate]) == pytest.approx(audio.rms(old), rel=0.1)


def test_movements_are_sample_exact_too(tmp_path, rate):
    journey = parse_journey("0:00 One.\n0:32 Two.\n1:36 Three.")
    timing = Timing(hop_s=32, context_s=32, tail_s=40, crossfade_s=4, blend_s=32)
    plan = make_plan("drone", 128, seed=3, bpm=60, key="D minor", timing=timing, journey=journey)
    paths = write_segments(tmp_path, plan, rate)
    total = sum(block.shape[0] for block in iter_piece(plan, paths))
    assert total == round(plan.duration_s * rate)


def test_crossfade_keeps_power_for_unrelated_signals(rate):
    noise = np.random.default_rng(0)
    outgoing = noise.normal(0, 0.1, (rate, 2)).astype(np.float32)
    incoming = noise.normal(0, 0.1, (rate, 2)).astype(np.float32)
    assert audio.rms(crossfade(outgoing, incoming)) == pytest.approx(0.1, rel=0.03)


def test_piece_is_sample_exact(tmp_path, rate, plan):
    paths = write_segments(tmp_path, plan, rate)
    total = sum(block.shape[0] for block in iter_piece(plan, paths))
    assert total == round(plan.duration_s * rate)


def test_joins_are_inaudible_for_a_continuous_signal(tmp_path, rate, plan):
    paths = write_segments(tmp_path, plan, rate)
    piece = np.concatenate(list(iter_piece(plan, paths)))
    steps = np.abs(np.diff(piece, axis=0))
    typical = np.percentile(steps, 99.9)
    for segment in plan.segments[1:]:
        join = int(segment.start_s * rate)
        assert steps[join - 8 : join + 8].max() <= 1.5 * typical


def test_partial_runs_can_be_previewed(tmp_path, rate, plan):
    paths = write_segments(tmp_path, plan, rate)[:2]
    total = sum(block.shape[0] for block in iter_piece(plan, paths))
    assert total == round(2 * plan.timing.hop_s * rate)


def test_rechunk_keeps_every_sample_in_order():
    data = np.arange(1000, dtype=np.float32)[:, None]
    pieces = [data[:7], data[7:300], data[300:301], data[301:]]
    chunks = list(rechunk(pieces, 128))
    assert all(chunk.shape[0] == 128 for chunk in chunks[:-1])
    assert np.array_equal(np.concatenate(chunks), data)


def test_soft_limit_leaves_quiet_audio_alone():
    block = np.full((8, 2), 0.3, dtype=np.float32)
    assert np.array_equal(soft_limit(block, 0.9), block)


def test_soft_limit_never_passes_the_ceiling():
    block = np.array([[5.0, -5.0], [0.95, -0.95]], dtype=np.float32)
    limited = soft_limit(block, 0.9)
    assert np.abs(limited).max() <= 0.9 + 1e-6


def test_level_ride_reaches_the_target():
    settings = MasterSettings(target_db=-20)
    gain = level_ride(np.full(600, 10 ** (-32 / 20)), settings)
    assert gain == pytest.approx(np.full(600, 12.0), abs=1e-6)


def test_level_ride_stays_within_its_limits():
    settings = MasterSettings(max_ride_db=6)
    levels = np.concatenate([np.full(600, 0.5), np.full(600, 0.0005)])
    gain = level_ride(levels, settings)
    assert gain.max() - gain.min() <= 12.0 + 1e-6


def test_level_ride_is_gentle():
    levels = np.concatenate([np.full(600, 0.05), np.full(600, 0.1)])
    gain = level_ride(levels, MasterSettings(ride_s=120))
    assert np.abs(np.diff(gain)).max() < 0.25


def test_master_writes_the_full_length_within_the_ceiling(tmp_path, rate, plan):
    paths = write_segments(tmp_path, plan, rate)
    outputs = render_master(plan, paths, tmp_path / "piece")
    assert [p.suffix for p in outputs] == [".wav"]
    info = sf.info(tmp_path / "piece.wav")
    assert info.frames == round(plan.duration_s * rate)
    assert info.samplerate == 48_000
    assert info.subtype == "PCM_24"
    assert info.format == "WAV"
    data, _ = sf.read(tmp_path / "piece.wav", dtype="float32")
    assert np.abs(data).max() <= 10 ** (-1 / 20) + 1e-4


def test_other_formats_can_be_written_too(tmp_path, rate, plan):
    paths = write_segments(tmp_path, plan, rate)
    outputs = render_master(plan, paths, tmp_path / "piece", formats=("wav", "flac", "mp3"))
    assert [p.suffix for p in outputs] == [".wav", ".flac", ".mp3"]
    assert sf.info(tmp_path / "piece.wav").frames == sf.info(tmp_path / "piece.wav").frames


def test_an_hour_fits_a_plain_wav():
    assert wav_container(3600 * 48_000) == "WAV"


def test_long_pieces_switch_to_rf64_before_the_wav_limit():
    assert wav_container(4 * 3600 * 48_000) == "WAV"
    assert wav_container(int(4.2 * 3600) * 48_000) == "RF64"


def test_rf64_files_are_written_and_read_back(tmp_path, tone, rate):
    path = tmp_path / "long.wav"
    with sf.SoundFile(path, "w", rate, 2, "PCM_24", format="RF64") as out:
        out.write(tone(1))
    assert sf.info(path).format == "RF64"
    assert sf.info(path).frames == rate


def test_master_fades_both_ends(tmp_path, rate, plan):
    paths = write_segments(tmp_path, plan, rate)
    render_master(plan, paths, tmp_path / "piece", MasterSettings(fade_in_s=4, fade_out_s=4),
                  formats=("wav",))
    data, _ = sf.read(tmp_path / "piece.wav", dtype="float32")
    assert np.abs(data[:100]).max() < 1e-3
    assert np.abs(data[-100:]).max() < 1e-3


def test_master_reaches_the_target_level(tmp_path, rate, plan):
    paths = write_segments(tmp_path, plan, rate)
    settings = MasterSettings(target_db=-24, fade_in_s=0, fade_out_s=0)
    render_master(plan, paths, tmp_path / "piece", settings, formats=("wav",))
    data, _ = sf.read(tmp_path / "piece.wav", dtype="float32")
    assert float(audio.db(audio.rms(data))) == pytest.approx(-24, abs=2.0)


def test_spiky_music_is_lowered_rather_than_limited_hard(tmp_path, rate, plan):
    paths = write_segments(tmp_path, plan, rate)
    for path in paths:
        data, _ = audio.read_region(path)
        data[:: rate // 2] = 0.9
        audio.write_audio(path, data * 0.2, rate)
    settings = MasterSettings(target_db=-10, fade_in_s=0, fade_out_s=0)
    render_master(plan, paths, tmp_path / "piece", settings, formats=("wav",))
    data, _ = sf.read(tmp_path / "piece.wav", dtype="float32")
    assert float(audio.db(audio.rms(data))) < -11


def test_unknown_format_is_rejected(tmp_path, rate, plan):
    paths = write_segments(tmp_path, plan, rate)
    with pytest.raises(ValueError):
        render_master(plan, paths, tmp_path / "piece", formats=("ogg",))
