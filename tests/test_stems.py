import numpy as np
import pytest
import soundfile as sf

from conftest import FAST_TIMING
from longdrive import audio, stems
from longdrive.engines import EngineError, Settings
from longdrive.plan import make_plan, parse_journey
from longdrive.run import Run, conduct, finish


def beats(rate, seconds, offset_s=0.0, bpm=60):
    t = np.arange(int(seconds * rate)) / rate
    phase = (t - offset_s) % (60 / bpm)
    wave = 0.5 * np.exp(-phase / 0.05) * np.sin(2 * np.pi * 60 * phase)
    return np.stack([wave, wave], axis=1).astype(np.float32)


def two_sections(tmp_path, rate, offsets=(0.1, 0.1), **levels):
    """A plan and a drum stem for it, each section with its beats at its own offset."""
    journey = parse_journey("0:00 One.\n1:04 Two.")
    plan = make_plan(
        "", 128, seed=1, bpm=60, key="A minor", timing=FAST_TIMING, journey=journey, **levels
    )
    first = beats(rate, 64 + 16, offsets[0])[: int(80 * rate)]
    second = beats(rate, 48, offsets[1])
    path = tmp_path / "drums.wav"
    audio.write_audio(path, np.concatenate([first, second]), rate)
    return plan, path


def test_the_kick_is_short_and_starts_and_ends_at_nothing(rate):
    kick = stems.soft_kick(rate)
    assert kick.size == pytest.approx(0.2 * rate)
    assert abs(kick[0]) < 1e-6
    assert abs(kick[-1]) < 1e-3


def test_a_kick_track_is_the_same_however_it_is_cut_into_blocks():
    kick = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    hits = np.array([1, 5, 8])
    whole = stems.kick_track(kick, hits, 0, 12)
    assert whole.tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 0]
    pieces = [stems.kick_track(kick, hits, start, 4) for start in (0, 4, 8)]
    assert np.concatenate(pieces).tolist() == whole.tolist()


def test_kicks_fall_on_every_beat_of_the_drums(tmp_path, rate):
    plan, path = two_sections(tmp_path, rate)
    times = stems.kick_times(plan, path, 128.0)
    assert np.diff(times) == pytest.approx(1.0, abs=1e-9)
    assert np.abs((times - 0.1 + 0.5) % 1.0 - 0.5).max() < 0.02
    assert times[0] < 1.2
    assert times[-1] > 126.5


def test_the_kick_eases_from_one_grid_to_the_next_without_a_stumble(tmp_path, rate):
    plan, path = two_sections(tmp_path, rate, offsets=(0.1, 0.4))
    times = stems.kick_times(plan, path, 128.0)
    assert np.abs((times[times < 60] - 0.1 + 0.5) % 1.0 - 0.5).max() < 0.02
    assert np.abs((times[times > 100] - 0.4 + 0.5) % 1.0 - 0.5).max() < 0.02
    assert np.abs(np.diff(times) - 1.0).max() < 0.02


def test_a_section_with_no_drums_keeps_the_pulse_going(tmp_path, rate):
    plan, path = two_sections(tmp_path, rate, offsets=(0.1, 0.4))
    stem, _ = audio.read_region(path)
    stem[int(64 * rate) :] = 0.0
    audio.write_audio(path, stem, rate)
    times = stems.kick_times(plan, path, 128.0)
    assert np.abs(np.diff(times) - 1.0).max() < 1e-9
    assert np.abs((times - 0.1 + 0.5) % 1.0 - 0.5).max() < 0.02


def test_a_piece_with_no_drums_at_all_still_gets_an_even_pulse(tmp_path, rate):
    plan, path = two_sections(tmp_path, rate)
    audio.write_audio(path, np.zeros((128 * rate, 2), dtype=np.float32), rate)
    times = stems.kick_times(plan, path, 128.0)
    assert times.size == pytest.approx(128, abs=1)
    assert np.abs(np.diff(times) - 1.0).max() < 1e-9


def test_the_swap_takes_out_the_stem_and_leaves_the_rest(tmp_path, rate, tone):
    music = tone(4.0)
    drums = beats(rate, 4.0)
    path = tmp_path / "drums.wav"
    audio.write_audio(path, drums, rate)
    swap = stems.DrumSwap(path, -120.0, np.zeros(10, np.float32), np.zeros(0, np.int64))
    blocks = [(music + drums)[:rate], (music + drums)[rate:]]
    out = np.concatenate(list(swap.apply(blocks)))
    assert out == pytest.approx(music, abs=1e-4)


def test_the_swap_can_leave_some_of_the_drums_in(tmp_path, rate, tone):
    drums = beats(rate, 4.0)
    path = tmp_path / "drums.wav"
    audio.write_audio(path, drums, rate)
    swap = stems.DrumSwap(path, -6.0, np.zeros(10, np.float32), np.zeros(0, np.int64))
    out = np.concatenate(list(swap.apply([drums])))
    assert out == pytest.approx(drums * 10 ** (-6 / 20), abs=1e-4)


@pytest.fixture
def fake_split(monkeypatch):
    """Stand in for Demucs: the drum stem is a fixed fraction of the piece."""
    calls = []

    def split(piece_path, drums_path):
        calls.append(piece_path)
        data, rate = sf.read(piece_path, dtype="float32", always_2d=True)
        audio.write_audio(drums_path, data * 0.5, rate)

    monkeypatch.setattr(stems, "split_drums", split)
    return calls


def swap_run(tmp_path, **levels):
    plan = make_plan(
        "drone", 64, seed=1, bpm=60, key="A minor", timing=FAST_TIMING, **levels
    )
    return Run.create(tmp_path / "piece", plan, "synth", Settings(), 1)


def test_a_finished_piece_has_its_drums_swapped(tmp_path, scripted, fake_split):
    run = swap_run(tmp_path, drums_db=-60, kick_db=-9)
    conduct(run, scripted())
    assert len(fake_split) == 1
    assert sf.info(run.stem.with_suffix(".wav")).duration == pytest.approx(64, abs=0.1)
    assert not (run.work_dir / "piece.wav").exists()
    assert (run.work_dir / "drums.wav").exists()


def test_a_second_master_needs_no_second_split(tmp_path, scripted, fake_split):
    run = swap_run(tmp_path, drums_db=-60)
    conduct(run, scripted())
    finish(run)
    assert len(fake_split) == 1


def test_a_piece_that_keeps_its_drums_is_never_split(tmp_path, scripted, fake_split):
    run = swap_run(tmp_path)
    conduct(run, scripted())
    assert fake_split == []
    assert not (run.work_dir / "drums.wav").exists()


def test_a_failed_split_leaves_the_run_able_to_retry(tmp_path, scripted, monkeypatch):
    def fail(piece_path, drums_path):
        raise EngineError("splitting the drums failed")

    monkeypatch.setattr(stems, "split_drums", fail)
    run = swap_run(tmp_path, kick_db=-9)
    with pytest.raises(EngineError):
        conduct(run, scripted())
    assert Run(run.directory).state["status"] == "stopped"
    assert len(Run(run.directory).committed_paths()) == len(run.plan.segments)
