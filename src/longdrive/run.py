"""Run a plan to completion: resumable, self-healing and safe to leave overnight."""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from longdrive import audio
from longdrive.engines import (
    Engine,
    EngineError,
    RenderRequest,
    Settings,
    create_engine,
)
from longdrive.keepawake import keep_awake, on_battery
from longdrive.master import MasterSettings, render_master
from longdrive.plan import Plan, Segment

RETRIES_PER_SEGMENT = 4
MIN_RENDER_TIMEOUT_S = 60 * 60
RENDER_TIMEOUT_FACTOR = 8
RESTART_BACKOFF_S = 10.0

STATE_FILE = "state.json"
PLAN_FILE = "plan.json"

log = logging.getLogger("longdrive")


class RunFailed(RuntimeError):
    """A segment could not be rendered. Finished work is safe and the run can be resumed."""


class Run:
    """A run directory: the plan, progress so far, and every committed segment."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.plan = Plan.load(directory / PLAN_FILE)
        self.state = json.loads((directory / STATE_FILE).read_text(encoding="utf-8"))

    @classmethod
    def create(
        cls, directory: Path, plan: Plan, engine: str, settings: Settings, candidates: int
    ) -> Run:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / PLAN_FILE).write_text(plan.to_json(), encoding="utf-8")
        state = {
            "status": "pending",
            "engine": engine,
            "settings": settings.to_dict(),
            "candidates": candidates,
            "segments": {},
        }
        _write_json(directory / STATE_FILE, state)
        return cls(directory)

    @property
    def settings(self) -> Settings:
        return Settings(**self.state["settings"])

    @property
    def character(self) -> audio.Character | None:
        saved = self.state.get("character")
        return audio.Character.from_dict(saved) if saved else None

    @property
    def work_dir(self) -> Path:
        return self.directory / "work"

    @property
    def stem(self) -> Path:
        return self.directory / self.directory.name

    def segment_path(self, index: int) -> Path:
        return self.directory / "segments" / f"{index:05d}.wav"

    def is_committed(self, index: int) -> bool:
        return str(index) in self.state["segments"] and self.segment_path(index).exists()

    def committed_paths(self) -> list[Path]:
        """Committed segments, stopping at the first gap so the audio stays contiguous."""
        paths = []
        for segment in self.plan.segments:
            if not self.is_committed(segment.index):
                break
            paths.append(self.segment_path(segment.index))
        return paths

    def save(self) -> None:
        self.state["updated_at"] = time.time()
        _write_json(self.directory / STATE_FILE, self.state)

    def timeout_for_next_render(self) -> float:
        """Generous enough for a slow GPU, tight enough to catch a real hang."""
        times = [s["render_s"] for s in self.state["segments"].values()]
        if not times:
            return MIN_RENDER_TIMEOUT_S
        return max(MIN_RENDER_TIMEOUT_S, RENDER_TIMEOUT_FACTOR * statistics.median(times))


@dataclass
class Candidate:
    audio: np.ndarray
    assessment: audio.Assessment
    seed: int
    render_s: float
    diagnostics: dict


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(f"{path.name}.partial")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def conduct(run: Run, engine: Engine | None = None, master: bool = True) -> list[Path]:
    """Render every outstanding segment, then master the piece."""
    engine = engine or create_engine(
        run.state["engine"], run.settings, log_path=run.directory / "engine.log"
    )
    if on_battery():
        log.warning("Running on battery: plug in, or Windows may throttle or hibernate this run.")

    run.work_dir.mkdir(exist_ok=True)
    run.state["status"] = "rendering"
    run.state.pop("last_error", None)
    run.save()
    outputs: list[Path] = []
    try:
        with keep_awake():
            pending = [s for s in run.plan.segments if not run.is_committed(s.index)]
            if pending:
                log.info("%d of %d segments to render", len(pending), len(run.plan.segments))
                engine.start()
            for segment in pending:
                _render_segment(run, engine, segment)
            engine.stop()
            if master:
                log.info("Mastering")
                outputs = render_master(run.plan, run.committed_paths(), run.stem, MasterSettings())
                for path in outputs:
                    log.info("Wrote %s", path)
    except BaseException as error:
        run.state["status"] = "stopped"
        run.state["last_error"] = str(error) or type(error).__name__
        run.save()
        raise
    finally:
        engine.stop()

    run.state["status"] = "complete"
    run.save()
    return outputs


def _render_segment(run: Run, engine: Engine, segment: Segment) -> None:
    timing = run.plan.timing
    context = None if segment.fresh else _write_context(run, segment)
    previous = audio.read_region(context)[0] if context else None
    required_s = segment.new_s + segment.overlap_s
    # A new movement is judged on its own terms: only its level has to match the piece.
    character = None if segment.fresh else run.character
    outgoing = _outgoing_audio(run, segment) if segment.fresh else None

    wanted = run.state["candidates"]
    limit = wanted + RETRIES_PER_SEGMENT
    candidates: list[Candidate] = []
    failures: list[str] = []
    attempt = 0

    while attempt < limit and sum(c.assessment.ok for c in candidates) < wanted:
        seed = segment.seed + attempt
        out_path = run.work_dir / f"candidate-{segment.index:05d}-{attempt}.wav"
        attempt += 1

        request = RenderRequest(
            out_path=out_path,
            caption=segment.prompt,
            seed=seed,
            new_s=segment.new_s + timing.tail_s,
            bpm=run.plan.bpm,
            key=segment.key,
            time_signature=run.plan.time_signature,
            piece_start_s=segment.start_s,
            context_path=context,
            context_s=timing.context_s if context else 0.0,
        )
        began = time.monotonic()
        try:
            diagnostics = engine.render(request, timeout_s=run.timeout_for_next_render()) or {}
        except EngineError as error:
            failures.append(str(error))
            log.warning("Segment %d attempt %d failed: %s", segment.index, attempt, error)
            _restart(engine)
            continue
        render_s = time.monotonic() - began

        new_audio, rate = audio.read_region(out_path, start_s=request.context_s)
        out_path.unlink(missing_ok=True)
        if outgoing is not None:
            new_audio = new_audio[audio.beat_offset(outgoing, new_audio, rate, run.plan.bar_s) :]
        # Judge everything that will be heard, including any overlap with the next window,
        # but no further: the tail is where the model fades out.
        kept = new_audio[: int(required_s * rate)]
        assessment = audio.assess(kept, rate, previous, character)
        if new_audio.shape[0] < int(required_s * rate):
            assessment.problems.append(f"short render ({new_audio.shape[0] / rate:.1f}s)")
        if not assessment.ok:
            log.warning("Segment %d seed %d: %s", segment.index, seed, assessment.problems)
        candidates.append(Candidate(new_audio, assessment, seed, render_s, diagnostics))

    if not candidates:
        raise RunFailed(f"segment {segment.index} failed {attempt} times: {failures[-1]}")

    best = min(candidates, key=lambda c: (not c.assessment.ok, c.assessment.score))
    piece_level = run.character.level_db if run.character else None
    if character is None:
        # The first minute alone is an intro. Reading further gives a steadier reference,
        # stopping short of the end of the render, where the model fades out.
        span = int((segment.new_s + timing.tail_s / 2) * rate)
        character = audio.Character.measure(best.audio[:span], rate)
        if piece_level is not None:
            best.assessment.trim_db = audio.trim_to(character.level_db, piece_level)
            character = replace(character, level_db=piece_level)
        run.state["character"] = character.to_dict()
    trimmed = best.audio * 10 ** (best.assessment.trim_db / 20)
    audio.write_audio(run.segment_path(segment.index), trimmed[: int(required_s * rate)], rate)
    run.state["segments"][str(segment.index)] = {
        "seed": best.seed,
        "attempts": attempt,
        "render_s": round(sum(c.render_s for c in candidates), 1),
        "assessment": best.assessment.to_dict(),
        **{key: value for key, value in best.diagnostics.items() if key != "event"},
    }
    run.save()
    if context:
        context.unlink(missing_ok=True)
    _report(run, segment, best)


def _outgoing_audio(run: Run, segment: Segment) -> np.ndarray:
    """The audio a new movement is mixed against, which its beats must line up with."""
    previous = run.plan.segments[segment.index - 1]
    return audio.read_region(
        run.segment_path(previous.index),
        start_s=previous.new_s,
        stop_s=previous.new_s + previous.overlap_s,
    )[0]


def _write_context(run: Run, segment: Segment) -> Path | None:
    """Save the end of the piece so far, for the next window to continue from."""
    if segment.index == 0:
        return None
    previous = run.plan.segments[segment.index - 1]
    timing = run.plan.timing
    tail, rate = audio.read_region(
        run.segment_path(previous.index),
        start_s=previous.new_s - timing.context_s,
        stop_s=previous.new_s,
    )
    path = run.work_dir / f"context-{segment.index:05d}.wav"
    audio.write_audio(path, tail, rate)
    return path


def _restart(engine: Engine) -> None:
    """Rebuild the engine after a failure, waiting out a transient driver problem.

    A failed restart is not final: the next attempt fails fast and restarts again,
    until the segment runs out of attempts.
    """
    engine.stop()
    time.sleep(RESTART_BACKOFF_S)
    try:
        engine.start()
    except EngineError as error:
        log.warning("Engine restart failed: %s", error)


def _report(run: Run, segment: Segment, best: Candidate) -> None:
    finished = run.state["segments"]
    spent = sum(s["render_s"] for s in finished.values())
    remaining = len(run.plan.segments) - len(finished)
    eta = spent / len(finished) * remaining
    check = best.assessment
    log.info(
        "[%d/%d] %s of music, %s to go | busyness %.2f, drift %.1f dB, dip %.1f dB",
        len(finished),
        len(run.plan.segments),
        _clock(segment.start_s + segment.new_s),
        _clock(eta),
        check.busyness,
        check.drift_db,
        check.dip_db,
    )


def _journey(plan: Plan) -> str:
    keys = len({s.key for s in plan.segments})
    return f"{plan.movements} movements, {keys} keys" if plan.movements > 1 else "steady"


def describe(run: Run) -> str:
    """Human summary of how a run is getting on."""
    plan, state = run.plan, run.state
    done = state["segments"]
    spent = sum(s["render_s"] for s in done.values())
    lines = [
        f"{run.directory.name}: {state['status']}",
        f"  prompt   {plan.prompt}",
        f"  music    {plan.bpm} bpm, {plan.key}, seed {plan.seed}",
        f"  journey  {_journey(plan)}",
        f"  length   {_clock(plan.duration_s)}",
        f"  progress {len(done)}/{len(plan.segments)} segments, {_clock(spent)} rendering",
    ]
    if done and len(done) < len(plan.segments):
        eta = spent / len(done) * (len(plan.segments) - len(done))
        lines.append(f"  eta      {_clock(eta)}")
    flagged = [i for i, s in done.items() if s["assessment"]["problems"]]
    if flagged:
        lines.append(f"  flagged  segments {', '.join(sorted(flagged, key=int))}")
    if state.get("last_error"):
        lines.append(f"  error    {state['last_error']}")
    return "\n".join(lines)
