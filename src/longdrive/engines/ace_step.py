"""Drive ACE-Step 1.5 in its own process and environment.

The model pins its own torch, transformers and gradio versions, so it lives in a
separate checkout with a separate virtualenv. Keeping it in a child process also
means an out-of-memory error or a driver hang costs one restart, not the whole run.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

from longdrive.engines.base import EngineError, EngineFatalError, RenderRequest, Settings

ENGINE_DIR = Path(__file__).resolve().parents[3] / "engines" / "ace-step"
WORKER = Path(__file__).with_name("ace_step_worker.py")

REPOSITORY = "https://github.com/ace-step/ACE-Step-1.5.git"
COMMIT = "ca1e85fe9430179831e6bc6be790c332190a3866"
WEIGHTS_REPO = "ACE-Step/Ace-Step1.5"
WEIGHTS_REVISION = "19671f406d603126926c1b7e2adc169acbcade22"
WEIGHTS_PATTERNS = ("acestep-v15-turbo/*", "vae/*", "Qwen3-Embedding-0.6B/*", "config.json")
DEMUCS_VERSION = "4.1.0"

# Larger DiT checkpoints published as separate repositories, at pinned revisions.
EXTRA_MODELS = {
    "acestep-v15-sft": (
        "ACE-Step/acestep-v15-sft",
        "c410d249e71ea9385a7b586865e65b1473e1098d",
    ),
}

LOAD_TIMEOUT_S = 30 * 60

# Host memory climbs by about 5 GB per render on Windows, where upstream's heap trim is
# a no-op, and exhausted the page file by the eighth render here. Past a first render
# the machine starts paging, which slowed sft renders from 150 s to 240 s, far more than
# the 35 s a reload costs. So every render gets a clean process.
RENDERS_PER_WORKER = 1


def engine_python() -> Path:
    scripts = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    return ENGINE_DIR / ".venv" / scripts


def missing_parts(settings: Settings) -> list[str]:
    """What still needs installing for these settings; empty means ready."""
    missing = []
    if not engine_python().exists():
        missing.append("engine environment")
    checkpoints = ENGINE_DIR / "checkpoints"
    for part in (settings.model, "vae", "Qwen3-Embedding-0.6B"):
        if not (checkpoints / part).is_dir():
            missing.append(f"weights: {part}")
    return missing


class AceStepEngine:
    def __init__(self, settings: Settings, log_path: Path | None = None) -> None:
        self.settings = settings
        self.log_path = log_path
        self._process: subprocess.Popen | None = None
        self._replies: queue.Queue[dict | None] = queue.Queue()
        self._log = None
        self._renders = 0
        self._has_loaded = False

    def start(self) -> None:
        if self._process is not None:
            return
        missing = missing_parts(self.settings)
        if missing:
            raise EngineFatalError(
                f"ACE-Step is not installed ({', '.join(missing)}). Run: longdrive setup"
            )

        environment = dict(
            os.environ,
            PYTHONUNBUFFERED="1",
            PYTHONIOENCODING="utf-8",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
        )

        self._log = open(self.log_path, "ab") if self.log_path else subprocess.DEVNULL  # noqa: SIM115
        self._replies = queue.Queue()
        self._renders = 0
        self._process = subprocess.Popen(
            [str(engine_python()), str(WORKER), str(ENGINE_DIR), self.settings.model],
            cwd=ENGINE_DIR,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
            encoding="utf-8",
        )
        threading.Thread(
            target=self._read_replies, args=(self._process, self._replies), daemon=True
        ).start()

        reply = self._await(LOAD_TIMEOUT_S, "model load")
        if reply.get("event") != "ready":
            self.stop()
            # A run reloads constantly, so one failure after a good load is transient.
            failure = EngineError if self._has_loaded else EngineFatalError
            raise failure(f"ACE-Step failed to load: {reply.get('error')}")
        self._has_loaded = True

    def render(self, request: RenderRequest, timeout_s: float) -> dict:
        if self._process is None:
            raise EngineError("engine is not running")
        if self._renders >= RENDERS_PER_WORKER:
            self.stop()
            self.start()
        self._renders += 1
        message = {**request.to_payload(), **self.settings.to_dict(), "op": "render"}
        try:
            self._process.stdin.write(json.dumps(message) + "\n")
            self._process.stdin.flush()
        except OSError as error:
            self.stop()
            raise EngineError(f"worker pipe closed: {error}") from error

        reply = self._await(timeout_s, "render")
        if reply.get("event") != "done":
            kind = "out of memory" if reply.get("oom") else "render failed"
            raise EngineError(f"{kind}: {reply.get('error')}")
        return reply

    def stop(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            try:
                process.stdin.write(json.dumps({"op": "quit"}) + "\n")
                process.stdin.flush()
                process.wait(timeout=15)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                process.kill()
                process.wait()
        self._close_log()

    def _close_log(self) -> None:
        if self._log not in (None, subprocess.DEVNULL):
            self._log.close()
        self._log = None

    @staticmethod
    def _read_replies(process: subprocess.Popen, replies: queue.Queue) -> None:
        """Each worker reports to its own queue, so a dying one can't confuse its successor."""
        for line in process.stdout:
            try:
                replies.put(json.loads(line))
            except json.JSONDecodeError:
                continue
        replies.put(None)

    def _await(self, timeout_s: float, what: str) -> dict:
        try:
            reply = self._replies.get(timeout=timeout_s)
        except queue.Empty:
            self._abandon()
            raise EngineError(f"{what} hung for {timeout_s:.0f}s; worker killed") from None
        if reply is None:
            code = self._abandon()
            raise EngineError(f"worker exited during {what} (code {code})")
        return reply

    def _abandon(self) -> int | None:
        process, self._process = self._process, None
        if process is None:
            return None
        process.kill()
        code = process.wait()
        self._close_log()
        return code


def install(settings: Settings, say=print) -> None:
    """Fetch the pinned engine and the weights these settings need. Safe to re-run."""
    if not (ENGINE_DIR / ".git").is_dir():
        say(f"Cloning ACE-Step into {ENGINE_DIR}")
        ENGINE_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", REPOSITORY, str(ENGINE_DIR)], check=True)
    subprocess.run(["git", "-C", str(ENGINE_DIR), "checkout", "-q", COMMIT], check=True)

    say("Syncing the engine environment (several GB on first run)")
    environment = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    subprocess.run(
        ["uv", "sync", "--frozen", "--python", "3.12"], cwd=ENGINE_DIR, env=environment, check=True
    )

    _download(WEIGHTS_REPO, WEIGHTS_REVISION, "checkpoints", list(WEIGHTS_PATTERNS), say)
    if settings.model in EXTRA_MODELS:
        repository, revision = EXTRA_MODELS[settings.model]
        _download(repository, revision, f"checkpoints/{settings.model}", None, say)

    # Syncing removes anything the engine's lockfile doesn't list, so this comes after it.
    say("Adding Demucs, which swaps a finished piece's drums")
    python = str(engine_python())
    subprocess.run(
        ["uv", "pip", "install", f"demucs=={DEMUCS_VERSION}", "--python", python],
        cwd=ENGINE_DIR, env=environment, check=True,
    )
    fetch = "from demucs.pretrained import get_model; get_model('htdemucs')"
    subprocess.run([python, "-c", fetch], cwd=ENGINE_DIR, check=True)


def _download(
    repository: str, revision: str, target: str, patterns: list[str] | None, say
) -> None:
    say(f"Fetching {repository} into {target}")
    script = (
        "import sys, json\n"
        "from huggingface_hub import snapshot_download\n"
        "repo, rev, target, patterns = json.loads(sys.argv[1])\n"
        "snapshot_download(repo_id=repo, revision=rev, local_dir=target,"
        " allow_patterns=patterns)\n"
    )
    arguments = json.dumps([repository, revision, target, patterns])
    subprocess.run([str(engine_python()), "-c", script, arguments], cwd=ENGINE_DIR, check=True)
