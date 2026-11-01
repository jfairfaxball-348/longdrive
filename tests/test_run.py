import numpy as np
import pytest
import soundfile as sf

from conftest import FAST_TIMING
from longdrive import audio
from longdrive import run as run_module
from longdrive.engines import EngineError, EngineFatalError, Settings
from longdrive.plan import make_plan, parse_journey
from longdrive.run import Run, RunFailed, conduct, describe


@pytest.fixture(autouse=True)
def no_restart_wait(monkeypatch):
    monkeypatch.setattr(run_module, "RESTART_BACKOFF_S", 0)


def test_full_run_produces_a_finished_piece(new_run, scripted):
    run = new_run(duration_s=80)
    outputs = conduct(run, scripted())
    assert run.state["status"] == "complete"
    assert sf.info(outputs[0]).frames == 80 * 48_000


def test_engines_are_given_a_directory_that_exists(new_run, scripted):
    """The real worker writes straight to out_path and does not create parents."""
    run = new_run(duration_s=20)

    class Strict(scripted):
        def render(self, request, timeout_s):
            assert request.out_path.parent.is_dir()
            return super().render(request, timeout_s)

    conduct(run, Strict())


def test_first_window_has_no_context_and_later_ones_do(new_run, scripted):
    run = new_run(duration_s=80)
    engine = scripted()
    conduct(run, engine)
    first, second, third = engine.requests
    assert first.context_path is None
    assert second.context_s == run.plan.timing.context_s
    assert third.piece_start_s == 2 * run.plan.timing.hop_s


def test_windows_over_render_past_the_hop(new_run, scripted):
    run = new_run(duration_s=50)
    engine = scripted()
    conduct(run, engine)
    assert engine.requests[0].new_s == run.plan.timing.hop_s + run.plan.timing.tail_s


def test_context_is_the_end_of_the_committed_audio(new_run, scripted, rate):
    run = new_run(duration_s=50)

    class Checking(scripted):
        def render(self, request, timeout_s):
            if request.context_path:
                context, _ = audio.read_region(request.context_path)
                index = round(request.piece_start_s / run.plan.timing.hop_s) - 1
                previous = run.plan.segments[index]
                stored, _ = audio.read_region(
                    run.segment_path(previous.index),
                    previous.new_s - run.plan.timing.context_s,
                    previous.new_s,
                )
                assert np.array_equal(context, stored)
                assert context.shape[0] == run.plan.timing.context_s * rate
            super().render(request, timeout_s)

    conduct(run, Checking())


def test_interrupted_run_resumes_without_rendering_again(new_run, scripted):
    run = new_run(duration_s=128)
    crashed = scripted({3: KeyboardInterrupt()})
    with pytest.raises(KeyboardInterrupt):
        conduct(run, crashed)
    assert run.state["status"] == "stopped"

    resumed = Run(run.directory)
    assert len(resumed.state["segments"]) == 2
    engine = scripted()
    conduct(resumed, engine)
    hop = run.plan.timing.hop_s
    assert [r.piece_start_s for r in engine.requests] == [2 * hop, 3 * hop]


def test_resumed_run_matches_an_uninterrupted_one(new_run, scripted, tmp_path):
    whole = new_run(duration_s=128, name="whole")
    conduct(whole, scripted())

    broken = new_run(duration_s=128, name="broken")
    with pytest.raises(KeyboardInterrupt):
        conduct(broken, scripted({3: KeyboardInterrupt()}))
    conduct(Run(broken.directory), scripted())

    a, _ = sf.read(whole.stem.with_suffix(".wav"))
    b, _ = sf.read(broken.stem.with_suffix(".wav"))
    assert np.array_equal(a, b)


def test_a_failed_render_is_retried_with_a_new_seed_and_restart(new_run, scripted):
    run = new_run(duration_s=20)
    engine = scripted({1: EngineError("out of memory")})
    conduct(run, engine)
    assert engine.requests[0].seed != engine.requests[1].seed
    assert engine.starts == 2
    assert run.state["status"] == "complete"


def test_a_segment_that_keeps_failing_stops_the_run_cleanly(new_run, scripted):
    run = new_run(duration_s=50)
    always = {n: EngineError("driver lost") for n in range(2, 40)}
    with pytest.raises(RunFailed, match="driver lost"):
        conduct(run, scripted(always))
    assert run.state["status"] == "stopped"
    assert "0" in run.state["segments"], "work already done must survive"


def test_a_failed_restart_does_not_end_the_run(new_run, scripted):
    run = new_run(duration_s=20)

    class RestartFailsOnce(scripted):
        def start(self):
            super().start()
            if self.starts == 2:
                raise EngineError("driver still resetting")

    engine = RestartFailsOnce({1: EngineError("device lost")})
    conduct(run, engine)
    assert run.state["status"] == "complete"
    assert engine.starts == 2


def test_unexpected_errors_leave_the_run_marked_stopped(new_run, scripted):
    run = new_run(duration_s=20)
    with pytest.raises(OSError):
        conduct(run, scripted({1: OSError("disk full")}))
    assert run.state["status"] == "stopped"
    assert "disk full" in run.state["last_error"]
    assert Run(run.directory).state["status"] == "stopped"


def test_a_resumed_run_clears_the_old_error(new_run, scripted):
    run = new_run(duration_s=20)
    with pytest.raises(OSError):
        conduct(run, scripted({1: OSError("disk full")}))
    conduct(Run(run.directory), scripted())
    assert "last_error" not in Run(run.directory).state


def test_fatal_engine_errors_are_not_retried(new_run, scripted):
    run = new_run(duration_s=20)
    engine = scripted({1: EngineFatalError("weights missing")})
    with pytest.raises(EngineFatalError):
        conduct(run, engine)
    assert engine.calls == 1


def test_best_candidate_is_kept(new_run, scripted):
    run = new_run(duration_s=40, candidates=2)

    class FirstIsLoud(scripted):
        def render(self, request, timeout_s):
            super().render(request, timeout_s)
            if request.context_path and self.calls % 2 == 1:
                data, rate = audio.read_region(request.out_path)
                cut = int(request.context_s * rate)
                data[cut:] *= 3
                audio.write_audio(request.out_path, data, rate)

    engine = FirstIsLoud()
    conduct(run, engine)
    assert run.state["segments"]["1"]["seed"] == engine.requests[3].seed


def test_committed_audio_is_trimmed_to_the_opening(new_run, scripted, rate):
    run = new_run(duration_s=50)

    class Loud(scripted):
        def render(self, request, timeout_s):
            super().render(request, timeout_s)
            if request.context_path:
                data, sample_rate = audio.read_region(request.out_path)
                data[int(request.context_s * sample_rate) :] *= 2
                audio.write_audio(request.out_path, data, sample_rate)

    conduct(run, Loud())
    assert run.state["segments"]["1"]["assessment"]["trim_db"] == pytest.approx(-6, abs=3)
    kept, _ = audio.read_region(run.segment_path(1))
    assert float(audio.db(audio.rms(kept))) == pytest.approx(run.character.level_db, abs=0.2)


def test_a_bad_render_is_replaced_when_possible(new_run, scripted):
    run = new_run(duration_s=50)

    class SilentOnce(scripted):
        def render(self, request, timeout_s):
            super().render(request, timeout_s)
            if self.calls == 2:
                data, rate = audio.read_region(request.out_path)
                data[int(request.context_s * rate) :] = 0
                audio.write_audio(request.out_path, data, rate)

    conduct(run, SilentOnce())
    assert run.state["segments"]["1"]["attempts"] == 2
    assert not run.state["segments"]["1"]["assessment"]["problems"]


def test_a_render_that_is_always_bad_is_kept_and_flagged(new_run, scripted):
    """Losing one segment should not throw away a night of rendering."""
    run = new_run(duration_s=50)

    class AlwaysQuiet(scripted):
        def render(self, request, timeout_s):
            super().render(request, timeout_s)
            if request.context_path:
                data, rate = audio.read_region(request.out_path)
                data[int(request.context_s * rate) :] = 0
                audio.write_audio(request.out_path, data, rate)

    conduct(run, AlwaysQuiet())
    assert run.state["status"] == "complete"
    assert run.state["segments"]["1"]["assessment"]["problems"]
    assert "flagged  segments 1" in describe(run)


def test_a_fade_in_the_discarded_tail_is_not_held_against_a_render(new_run, scripted):
    run = new_run(duration_s=50)

    class FadesAtTheEnd(scripted):
        def render(self, request, timeout_s):
            super().render(request, timeout_s)
            data, rate = audio.read_region(request.out_path)
            data[-10 * rate :] = 0
            audio.write_audio(request.out_path, data, rate)

    conduct(run, FadesAtTheEnd())
    assert all(s["attempts"] == 1 for s in run.state["segments"].values())


def test_a_steady_piece_keeps_its_opening_character(new_run, scripted):
    run = new_run(duration_s=80)
    seen = []

    class Watching(scripted):
        def render(self, request, timeout_s):
            seen.append(run.character)
            return super().render(request, timeout_s)

    conduct(run, Watching())
    assert seen[1] == seen[2] == run.character


def journey_run(tmp_path, text, duration_s=128):
    journey = parse_journey(text)
    plan = make_plan(
        "drone", duration_s, seed=1, bpm=60, key="A minor", timing=FAST_TIMING, journey=journey
    )
    return Run.create(tmp_path / "journey", plan, "synth", Settings(), 1)


def test_each_movement_starts_from_a_fresh_render(tmp_path, scripted, monkeypatch):
    monkeypatch.setattr(audio, "MAX_DIP_DB", 99.0)  # the synth's slow swells look like dips
    run = journey_run(tmp_path, "0:00 Start plainly.\n1:04 [C major] Open up.")
    engine = scripted()
    conduct(run, engine)
    assert [r.context_path is None for r in engine.requests] == [True, False, True, False]
    assert engine.requests[1].caption == "Start plainly."
    assert engine.requests[2].caption == "Open up."
    assert [r.key for r in engine.requests] == ["A minor", "A minor", "C major", "C major"]


def test_a_breakdown_in_the_blend_counts_against_a_render(tmp_path, scripted):
    """The outgoing tail is heard, so it has to be judged, though it lies past the hop."""
    run = journey_run(tmp_path, "0:00 One.\n1:04 Two.")
    hop = run.plan.timing.hop_s

    class EmptyTail(scripted):
        def render(self, request, timeout_s):
            super().render(request, timeout_s)
            if self.calls == 2:
                data, rate = audio.read_region(request.out_path)
                start = int((request.context_s + hop + 8) * rate)
                data[start : start + 16 * rate] *= 0.01
                audio.write_audio(request.out_path, data, rate)

    conduct(run, EmptyTail())
    assert run.state["segments"]["1"]["attempts"] > 1


def test_a_finished_journey_is_sample_exact(tmp_path, scripted):
    run = journey_run(tmp_path, "0:00 One.\n1:04 Two.")
    outputs = conduct(run, scripted())
    assert sf.info(outputs[0]).frames == 128 * 48_000


def test_a_new_movement_keeps_the_level_of_the_piece(tmp_path, scripted):
    run = journey_run(tmp_path, "0:00 One.\n1:04 Two.")

    class LouderLater(scripted):
        def render(self, request, timeout_s):
            super().render(request, timeout_s)
            if request.piece_start_s >= 64:
                data, rate = audio.read_region(request.out_path)
                data[int(request.context_s * rate) :] *= 3
                audio.write_audio(request.out_path, data, rate)

    conduct(run, LouderLater())
    first, _ = audio.read_region(run.segment_path(0), stop_s=32)
    later, _ = audio.read_region(run.segment_path(2), stop_s=32)
    assert audio.db(audio.rms(later)) == pytest.approx(audio.db(audio.rms(first)), abs=3)


def test_a_movement_ends_with_enough_audio_to_blend(tmp_path, scripted):
    run = journey_run(tmp_path, "0:00 One.\n1:04 Two.")
    conduct(run, scripted())
    last_of_first = audio.read_region(run.segment_path(1))[0]
    assert last_of_first.shape[0] == int((32 + run.plan.timing.blend_s) * 48_000)


def test_a_movement_resets_what_later_windows_are_held_to(tmp_path, scripted):
    run = journey_run(tmp_path, "0:00 One.\n1:04 Two.")
    seen = []

    class Watching(scripted):
        def render(self, request, timeout_s):
            seen.append(run.character)
            return super().render(request, timeout_s)

    conduct(run, Watching())
    assert seen[1] == seen[2]
    assert run.character.shape != seen[2].shape or run.character == seen[2]


def test_the_opening_sets_the_character_for_the_rest_of_the_piece(new_run, scripted):
    run = new_run(duration_s=50)
    conduct(run, scripted())
    saved = Run(run.directory).character
    assert saved == run.character
    assert len(saved.shape) == 24


def test_a_render_that_loses_the_character_is_replaced(new_run, scripted):
    run = new_run(duration_s=50)

    class BrightOnce(scripted):
        def render(self, request, timeout_s):
            super().render(request, timeout_s)
            if self.calls == 2:
                data, rate = audio.read_region(request.out_path)
                cut = int(request.context_s * rate)
                noise = np.random.default_rng(0).normal(0, 0.2, data[cut:].shape)
                data[cut:] += noise.astype(np.float32)
                audio.write_audio(request.out_path, data, rate)

    conduct(run, BrightOnce())
    assert run.state["segments"]["1"]["attempts"] == 2
    assert not run.state["segments"]["1"]["assessment"]["problems"]


def test_short_renders_are_treated_as_bad(new_run, scripted):
    run = new_run(duration_s=50)

    class ShortOnce(scripted):
        def render(self, request, timeout_s):
            super().render(request, timeout_s)
            if self.calls == 2:
                data, rate = audio.read_region(request.out_path)
                short = int((request.context_s + 8) * rate)
                audio.write_audio(request.out_path, data[:short], rate)

    conduct(run, ShortOnce())
    assert run.state["segments"]["1"]["attempts"] == 2


def test_timeout_grows_with_slow_renders_but_never_shrinks_below_the_floor(new_run):
    run = new_run()
    assert run.timeout_for_next_render() == run_module.MIN_RENDER_TIMEOUT_S
    run.state["segments"] = {"0": {"render_s": 10.0}}
    assert run.timeout_for_next_render() == run_module.MIN_RENDER_TIMEOUT_S
    run.state["segments"] = {"0": {"render_s": 3000.0}}
    assert run.timeout_for_next_render() == 8 * 3000.0


def test_only_contiguous_segments_are_used(new_run, scripted):
    run = new_run(duration_s=128)
    conduct(run, scripted())
    run.segment_path(1).unlink()
    assert len(run.committed_paths()) == 1


def test_describe_reports_progress(new_run, scripted):
    run = new_run(duration_s=80)
    with pytest.raises(KeyboardInterrupt):
        conduct(run, scripted({2: KeyboardInterrupt()}))
    text = describe(Run(run.directory))
    assert "1/3 segments" in text
    assert "eta" in text
