"""The supervisor's recovery paths, driven by a stand-in worker instead of the model."""

import sys
import textwrap

import pytest

from longdrive.engines import EngineError, EngineFatalError, RenderRequest, Settings, ace_step

STUB = textwrap.dedent(
    """
    import json, os, sys, time

    channel = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    print("libraries may print anything")

    def send(**message):
        channel.write(json.dumps(message) + "\\n")

    mode = sys.argv[2]
    loads = sys.argv[1] + "/loads"
    count = len(open(loads).read()) if os.path.exists(loads) else 0
    open(loads, "a").write("x")
    if mode == "fail-load" or (mode == "fail-second-load" and count == 1):
        send(event="failed", error="no weights")
        raise SystemExit(1)
    send(event="ready")

    for line in sys.stdin:
        request = json.loads(line)
        if request.get("op") == "quit":
            break
        if mode == "hang":
            time.sleep(60)
        if mode == "crash":
            os._exit(3)
        if mode == "oom":
            send(event="error", error="CUDA out of memory", oom=True)
            continue
        send(event="done", peak_vram_gb=1.5)
    """
)


@pytest.fixture
def engine(tmp_path, monkeypatch):
    stub = tmp_path / "stub_worker.py"
    stub.write_text(STUB, encoding="utf-8")
    monkeypatch.setattr(ace_step, "WORKER", stub)
    monkeypatch.setattr(ace_step, "engine_python", lambda: sys.executable)
    monkeypatch.setattr(ace_step, "missing_parts", lambda settings: [])
    monkeypatch.setattr(ace_step, "ENGINE_DIR", tmp_path)

    created = []

    def make(model: str) -> ace_step.AceStepEngine:
        instance = ace_step.AceStepEngine(Settings(model=model), log_path=tmp_path / "engine.log")
        created.append(instance)
        return instance

    yield make
    for instance in created:
        instance.stop()


@pytest.fixture
def request_(tmp_path):
    return RenderRequest(
        out_path=tmp_path / "out.wav",
        caption="c",
        seed=1,
        new_s=10,
        bpm=60,
        key="C major",
        time_signature="4",
    )


def test_a_healthy_worker_renders(engine, request_):
    worker = engine("ok")
    worker.start()
    assert worker.render(request_, timeout_s=10)["peak_vram_gb"] == 1.5
    worker.stop()


def test_stop_is_safe_to_repeat(engine):
    worker = engine("ok")
    worker.start()
    worker.stop()
    worker.stop()


def test_worker_output_goes_to_the_log(engine, tmp_path):
    worker = engine("ok")
    worker.start()
    worker.stop()
    assert "libraries may print anything" in (tmp_path / "engine.log").read_text()


def test_a_failed_load_is_fatal(engine):
    worker = engine("fail-load")
    with pytest.raises(EngineFatalError, match="no weights"):
        worker.start()


def test_a_failed_reload_can_be_retried(engine, request_):
    """Runs reload hundreds of times, so one bad load must not end the night."""
    worker = engine("fail-second-load")
    worker.start()
    worker.render(request_, timeout_s=10)
    with pytest.raises(EngineError, match="no weights"):
        worker.render(request_, timeout_s=10)
    worker.start()
    assert worker.render(request_, timeout_s=10)["peak_vram_gb"] == 1.5


def test_missing_install_is_fatal_with_instructions(engine, monkeypatch):
    monkeypatch.setattr(ace_step, "missing_parts", lambda settings: ["weights: vae"])
    with pytest.raises(EngineFatalError, match="longdrive setup"):
        engine("ok").start()


def test_a_hung_worker_is_killed(engine, request_):
    worker = engine("hang")
    worker.start()
    with pytest.raises(EngineError, match="hung"):
        worker.render(request_, timeout_s=0.5)
    assert worker._process is None


def test_a_crashed_worker_is_reported(engine, request_):
    worker = engine("crash")
    worker.start()
    with pytest.raises(EngineError, match="exited"):
        worker.render(request_, timeout_s=10)


def test_out_of_memory_is_named(engine, request_):
    worker = engine("oom")
    worker.start()
    with pytest.raises(EngineError, match="out of memory"):
        worker.render(request_, timeout_s=10)


def test_engine_restarts_after_a_crash(engine, request_):
    worker = engine("crash")
    worker.start()
    with pytest.raises(EngineError):
        worker.render(request_, timeout_s=10)
    worker.stop()
    worker.start()
    assert worker._process is not None


def test_workers_are_recycled_before_they_can_leak_to_death(engine, request_):
    worker = engine("ok")
    worker.start()
    first = worker._process.pid
    for _ in range(ace_step.RENDERS_PER_WORKER):
        worker.render(request_, timeout_s=10)
    assert worker._process.pid == first
    worker.render(request_, timeout_s=10)
    assert worker._process.pid != first


def test_a_finished_worker_never_reports_into_its_successor(engine, request_):
    """Each recycle races the old reader thread against the new worker's load."""
    worker = engine("ok")
    worker.start()
    for _ in range(ace_step.RENDERS_PER_WORKER * 12):
        worker.render(request_, timeout_s=10)


def test_rendering_before_start_is_an_engine_error(engine, request_):
    with pytest.raises(EngineError):
        engine("ok").render(request_, timeout_s=1)
