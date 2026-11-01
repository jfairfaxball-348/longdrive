"""Command line interface."""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

from longdrive import __version__
from longdrive.engines import EngineError, EngineFatalError, Settings
from longdrive.master import OUTPUT_FORMATS, render_master
from longdrive.plan import EVOLUTIONS, make_plan, parse_journey
from longdrive.run import Run, RunFailed, conduct, describe

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "runs"


# turbo stays in lock with the context, which suits a steady groove. sft has real
# classifier-free guidance, so it holds sparse music calmer, but it stops and restarts
# more and its windows drift apart.
MODELS = {
    "turbo": Settings(),
    "sft": Settings(model="acestep-v15-sft", steps=50, guidance=7.0),
}

CANDIDATES = {"draft": 1, "standard": 3, "best": 6}

_DURATION_RE = re.compile(r"^(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?$")


def parse_duration(text: str) -> float:
    """Accept 3600, 90s, 45m, 1.5h or 1h30m."""
    text = text.strip().lower()
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    match = _DURATION_RE.match(text)
    if not match or not any(match.groups()):
        raise argparse.ArgumentTypeError(f"not a duration: {text!r} (try 45m, 1h30m or 3600)")
    hours, minutes, seconds = (float(group or 0) for group in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def slug(text: str, limit: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:limit].rstrip("-") or "piece"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="longdrive", description="Long-form instrumental music from a text prompt."
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    setup = commands.add_parser("setup", help="install the ACE-Step engine and its weights")
    setup.add_argument("--model", choices=MODELS, default="turbo")

    generate = commands.add_parser("generate", help="render a new piece")
    generate.add_argument(
        "prompt", nargs="?", default="",
        help='what it should sound like, e.g. "warm piano and pads"',
    )
    generate.add_argument(
        "-d", "--duration", type=parse_duration, default=3600.0,
        help="length such as 45m, 1h30m or 3600 (default 1h)",
    )
    generate.add_argument(
        "-q", "--quality", choices=CANDIDATES, default="standard",
        help="how many renders to choose from per window",
    )
    generate.add_argument(
        "-m", "--model", choices=MODELS, default="turbo",
        help="turbo for grooves, sft for sparse music without drums",
    )
    generate.add_argument(
        "-e", "--evolve", choices=EVOLUTIONS, default="gentle",
        help="how far the piece travels over its length",
    )
    generate.add_argument(
        "--journey", type=Path, help="a timeline of prompts, instead of --evolve (see the README)"
    )
    generate.add_argument("--bpm", type=int, help="tempo (default: from the prompt, else slow)")
    generate.add_argument("--key", help='musical key, e.g. "D minor"')
    generate.add_argument("--seed", type=int, help="repeat a previous run exactly")
    generate.add_argument("--name", help="run directory name")
    generate.add_argument("--engine", choices=("ace-step", "synth"), default="ace-step")
    generate.add_argument("--dry-run", action="store_true", help="print the plan and stop")

    resume = commands.add_parser("resume", help="continue an interrupted run")
    resume.add_argument("run", type=Path)

    status = commands.add_parser("status", help="show progress of a run")
    status.add_argument("run", type=Path, nargs="?")

    master = commands.add_parser(
        "master", help="write the piece from the segments finished so far"
    )
    master.add_argument("run", type=Path)
    master.add_argument(
        "-f", "--format", nargs="+", choices=OUTPUT_FORMATS, default=["wav"],
        help="file formats to write (default wav)",
    )

    return parser


def latest_run() -> Path | None:
    runs = [path for path in RUNS_DIR.glob("*") if (path / "plan.json").exists()]
    return max(runs, key=lambda path: path.stat().st_mtime, default=None)


def resolve_run(path: Path | None) -> Path:
    resolved = path or latest_run()
    if resolved is None:
        raise SystemExit("no runs yet")
    if not (resolved / "plan.json").exists():
        resolved = RUNS_DIR / resolved
    if not (resolved / "plan.json").exists():
        raise SystemExit(f"not a run directory: {path}")
    return resolved


def configure_logging(run: Run) -> None:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(run.directory / "run.log", encoding="utf-8"),
    ]
    logger = logging.getLogger("longdrive")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def run_to_completion(run: Run) -> int:
    configure_logging(run)
    print(describe(run))
    try:
        conduct(run)
    except KeyboardInterrupt:
        print(f"\nStopped. Continue with: longdrive resume {run.directory}")
        return 130
    except (RunFailed, EngineError, EngineFatalError) as error:
        print(f"\nRun stopped: {error}\nContinue with: longdrive resume {run.directory}")
        return 1
    print(describe(run))
    return 0


def command_generate(args: argparse.Namespace) -> int:
    if not args.prompt and not args.journey:
        raise SystemExit("give a prompt, or a timeline with --journey")
    plan = make_plan(
        args.prompt,
        args.duration,
        seed=args.seed,
        bpm=args.bpm,
        key=args.key,
        evolution=EVOLUTIONS[args.evolve],
        journey=parse_journey(args.journey.read_text(encoding="utf-8")) if args.journey else None,
    )
    if args.dry_run:
        print(plan.to_json())
        return 0

    name = args.name or f"{time.strftime('%Y%m%d-%H%M')}-{slug(plan.prompt)}"
    if (RUNS_DIR / name).exists():
        raise SystemExit(f"run {name} already exists; continue it with: longdrive resume {name}")
    run = Run.create(
        RUNS_DIR / name, plan, args.engine, MODELS[args.model], CANDIDATES[args.quality]
    )
    return run_to_completion(run)


def command_setup(args: argparse.Namespace) -> int:
    from longdrive.engines.ace_step import install, missing_parts

    settings = MODELS[args.model]
    install(settings)
    missing = missing_parts(settings)
    if missing:
        print(f"Still missing: {', '.join(missing)}")
        return 1
    print("Ready.")
    return 0


def command_master(args: argparse.Namespace) -> int:
    run = Run(resolve_run(args.run))
    paths = run.committed_paths()
    if not paths:
        raise SystemExit("no finished segments yet")
    for path in render_master(run.plan, paths, run.stem, formats=tuple(args.format)):
        print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    # Redirected output on Windows uses a legacy code page that can't print every prompt.
    sys.stdout.reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    if args.command == "setup":
        return command_setup(args)
    if args.command == "generate":
        return command_generate(args)
    if args.command == "resume":
        return run_to_completion(Run(resolve_run(args.run)))
    if args.command == "status":
        print(describe(Run(resolve_run(args.run))))
        return 0
    if args.command == "master":
        return command_master(args)
    raise AssertionError(args.command)
