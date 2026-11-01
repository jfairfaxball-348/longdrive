import numpy as np
import pytest

from longdrive import audio


def test_write_and_read_keep_peaks_above_full_scale(tmp_path, rate):
    loud = np.full((rate, 2), 1.5, dtype=np.float32)
    path = tmp_path / "loud.wav"
    audio.write_audio(path, loud, rate)
    back, _ = audio.read_region(path)
    assert back.max() == pytest.approx(1.5)


def test_write_leaves_no_partial_file(tmp_path, tone, rate):
    path = tmp_path / "a.wav"
    audio.write_audio(path, tone(1), rate)
    assert [p.name for p in tmp_path.iterdir()] == ["a.wav"]


def test_read_region_is_sample_accurate(tmp_path, rate):
    ramp = np.arange(rate * 3, dtype=np.float32)[:, None] / (rate * 3)
    path = tmp_path / "ramp.wav"
    audio.write_audio(path, np.repeat(ramp, 2, axis=1), rate)
    region, _ = audio.read_region(path, start_s=1.0, stop_s=2.0)
    assert region.shape == (rate, 2)
    assert region[0, 0] == pytest.approx(1 / 3, abs=1e-6)


def test_mono_becomes_stereo():
    assert audio.to_stereo(np.zeros((10, 1))).shape == (10, 2)


def test_rms_of_a_sine_is_peak_over_root_two(tone):
    assert audio.rms(tone(1, level=0.5)) == pytest.approx(0.5 / np.sqrt(2), rel=1e-3)


def test_continuation_of_the_same_tone_is_clean(tone, rate):
    whole = tone(12)
    before, after = whole[: 6 * rate], whole[6 * rate :]
    result = audio.assess(after, rate, before)
    assert result.ok
    assert result.level_jump_db < 0.1
    assert result.timbre_jump_db < 0.5


def test_flags_dead_air(tone, rate):
    gap = tone(10)
    gap[3 * rate : 9 * rate] = 0
    assert any("dead air" in p for p in audio.assess(gap, rate).problems)


def test_tolerates_a_short_quiet_passage(tone, rate):
    gap = tone(10)
    gap[3 * rate : 5 * rate] = 0
    assert audio.assess(gap, rate).ok


def test_flags_non_finite_output(tone, rate):
    broken = tone(5)
    broken[100, 0] = np.nan
    assert any("non-finite" in p for p in audio.assess(broken, rate).problems)


def test_flags_a_level_jump(tone, rate):
    result = audio.assess(tone(5, level=0.5), rate, tone(5, level=0.02))
    assert any("level jump" in p for p in result.problems)


def test_a_breakdown_is_caught(rate):
    noise = np.random.default_rng(0).normal(0, 0.1, (60 * rate, 2)).astype(np.float32)
    assert audio.assess(noise, rate).ok
    noise[24 * rate : 40 * rate] *= 0.1
    result = audio.assess(noise, rate)
    assert result.dip_db == pytest.approx(20, abs=1)
    assert any("breakdown" in p for p in result.problems)


def test_emptier_than_the_opening_scores_worse(rate):
    noise = np.random.default_rng(0)
    steady = noise.normal(0, 0.1, (10 * rate, 2)).astype(np.float32)
    beat = np.sign(np.sin(2 * np.pi * 2 * np.arange(10 * rate) / rate))
    pulses = steady * (1 + 0.9 * beat)[:, None]
    opening = audio.Character.measure(pulses.astype(np.float32), rate)
    busy_enough = audio.assess(pulses.astype(np.float32), rate, character=opening)
    emptier = audio.assess(steady, rate, character=opening)
    assert emptier.score > busy_enough.score


def kicks(rate, seconds, offset_s=0.0, bpm=120):
    """A plain four-on-the-floor kick pattern, starting ``offset_s`` late."""
    t = np.arange(int(seconds * rate)) / rate
    beat = 60 / bpm
    phase = (t - offset_s) % beat
    wave = 0.5 * np.exp(-phase / 0.08) * np.sin(2 * np.pi * 60 * phase)
    return np.stack([wave, wave], axis=1).astype(np.float32)


@pytest.mark.parametrize("offset_s", [0.0, 0.11, 0.25, 0.37])
def test_the_beat_is_found_to_within_a_few_milliseconds(rate, offset_s):
    found = audio.beat_phase(kicks(rate, 20, offset_s=offset_s), rate, 0.5)
    assert (found - offset_s + 0.25) % 0.5 - 0.25 == pytest.approx(0.0, abs=0.015)


@pytest.mark.parametrize("offset_s", [0.0, 0.11, 0.25, 0.37])
def test_beats_are_lined_up_to_within_a_few_milliseconds(rate, offset_s):
    bar_s = 2.0
    outgoing = kicks(rate, 16)
    incoming = kicks(rate, 20, offset_s=offset_s)
    shift = audio.beat_offset(outgoing, incoming, rate, bar_s)
    beat_frames = rate // 2
    remainder = (shift - offset_s * rate) % beat_frames
    assert min(remainder, beat_frames - remainder) <= 0.01 * rate


def test_a_short_incoming_render_is_left_alone(rate):
    assert audio.beat_offset(kicks(rate, 16), kicks(rate, 1), rate, 2.0) == 0


def test_a_timbre_change_at_the_join_is_measured_but_is_not_a_failure(tone, rate):
    same = audio.assess(tone(5, hz=220), rate, tone(5, hz=220))
    moved = audio.assess(tone(5, hz=3000), rate, tone(5, hz=220))
    assert moved.ok
    assert moved.timbre_jump_db > same.timbre_jump_db + 5


def test_slow_swells_do_not_read_as_a_jump(rate):
    t = np.arange(120 * rate) / rate
    swell = 10 ** (6 * np.sin(2 * np.pi * t / 40) / 20)
    wave = (0.1 * swell * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    whole = np.stack([wave, wave], axis=1)
    middle = 40 * rate
    assert audio.assess(whole[middle:], rate, whole[:middle]).ok


def test_a_new_chord_on_a_bar_line_is_not_a_jump(rate):
    """Struck notes decay, so a join sits between a quiet moment and a loud one."""
    bar = 4 * rate
    decay = np.exp(-np.arange(bar) / (0.6 * rate))
    bars = np.tile(decay, 4)
    tone = np.sin(2 * np.pi * 220 * np.arange(bars.size) / rate)
    wave = (0.4 * bars * tone).astype(np.float32)
    whole = np.stack([wave, wave], axis=1)
    assert audio.assess(whole[2 * bar :], rate, whole[: 2 * bar]).ok


def test_drift_catches_the_arrival_of_bright_percussion(tone, rate):
    calm = tone(10, hz=200)
    opening = audio.Character.measure(calm, rate)
    noise = np.random.default_rng(0).normal(0, 0.05, calm.shape).astype(np.float32)
    assert audio.assess(tone(10, hz=200), rate, character=opening).ok
    result = audio.assess(calm + noise, rate, character=opening)
    assert any("drift" in p for p in result.problems)


def test_drift_ignores_overall_loudness(rate):
    noise = np.random.default_rng(0).normal(0, 0.1, (10 * rate, 2)).astype(np.float32)
    opening = audio.Character.measure(noise, rate)
    assert audio.assess(noise * 0.25, rate, character=opening).drift_db < 0.1


def test_a_louder_render_is_trimmed_rather_than_rejected(rate):
    noise = np.random.default_rng(0).normal(0, 0.03, (10 * rate, 2)).astype(np.float32)
    opening = audio.Character.measure(noise, rate)
    result = audio.assess(noise * 2, rate, character=opening)
    assert result.trim_db == pytest.approx(-6, abs=0.1)
    assert result.ok


def test_a_quieter_render_is_trimmed_up(rate):
    noise = np.random.default_rng(0).normal(0, 0.1, (10 * rate, 2)).astype(np.float32)
    opening = audio.Character.measure(noise, rate)
    assert audio.assess(noise * 0.5, rate, character=opening).trim_db == pytest.approx(6, abs=0.1)


def test_trim_has_limits(rate):
    noise = np.random.default_rng(0).normal(0, 0.1, (10 * rate, 2)).astype(np.float32)
    opening = audio.Character.measure(noise, rate)
    assert audio.assess(noise * 0.01, rate, character=opening).trim_db == audio.MAX_TRIM_DB


def test_the_join_is_judged_after_trimming(rate):
    noise = np.random.default_rng(0).normal(0, 0.03, (10 * rate, 2)).astype(np.float32)
    opening = audio.Character.measure(noise, rate)
    assert audio.assess(noise * 3, rate, noise, opening).level_jump_db < 0.1


def test_a_louder_render_scores_worse_all_the_same(rate):
    noise = np.random.default_rng(0).normal(0, 0.03, (10 * rate, 2)).astype(np.float32)
    opening = audio.Character.measure(noise, rate)
    same = audio.assess(noise, rate, character=opening)
    louder = audio.assess(noise * 2, rate, character=opening)
    assert louder.score > same.score


def test_the_calmest_opening_scores_best(tone, rate):
    held = tone(8)
    struck = tone(8) * np.exp(-np.arange(8 * rate) % (rate // 2) / (rate / 20))[:, None]
    assert audio.assess(held, rate).score < audio.assess(struck.astype(np.float32), rate).score


def test_staying_calm_beats_a_cleaner_join(tone, rate):
    """Holding the character matters more than the neatest possible join."""
    previous = tone(8)
    opening = audio.Character.measure(previous, rate)
    held = tone(8) * 0.8
    struck = tone(8) * np.exp(-np.arange(8 * rate) % (rate // 2) / (rate / 20))[:, None]
    calm = audio.assess(held, rate, previous, opening)
    busy = audio.assess(struck.astype(np.float32), rate, previous, opening)
    assert calm.score < busy.score


def test_an_opening_survives_being_saved(tone, rate):
    opening = audio.Character.measure(tone(4), rate)
    assert audio.Character.from_dict(opening.to_dict()) == opening


def test_busyness_separates_sustained_from_busy_audio(tone, rate):
    held = tone(8)
    struck = tone(8) * np.exp(-np.arange(8 * rate) % (rate // 2) / (rate / 20))[:, None]
    assert audio.busyness(struck.astype(np.float32), rate) > 3 * audio.busyness(held, rate)


def test_band_levels_find_the_right_band(tone, rate):
    low, high = audio.band_levels(tone(2, hz=100), rate), audio.band_levels(tone(2, hz=8000), rate)
    assert low.argmax() < 8 < high.argmax()
