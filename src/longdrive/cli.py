"""Command line interface."""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

from longdrive import __version__
from longdrive.engines import EngineError, EngineFatalError, Settings
from longdrive.master import OUTPUT_FORMATS
from longdrive.plan import (
    EVOLUTIONS,
    Evolution,
    Plan,
    WordList,
    journey_text,
    make_plan,
    parse_journey,
    parse_wordlist,
)
from longdrive.run import Run, RunFailed, conduct, describe, finish

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "runs"
WORDLISTS_DIR = ROOT / "wordlists"

# A piece from a word list starts a new section this often, unless told otherwise.
WORD_SECTION_S = 210.0


# turbo stays in lock with the context, which suits a steady groove. sft has real
# classifier-free guidance, so it holds sparse music calmer, but it stops and restarts
# more and its windows drift apart.
MODELS = {
    "turbo": Settings(),
    "sft": Settings(model="acestep-v15-sft", steps=50, guidance=7.0),
}

CANDIDATES = {"draft": 1, "standard": 3, "best": 6}

DRUMS_HELP = "level of the model's drums in dB, split out with Demucs (-60 takes them out)"
KICK_HELP = "add a plain kick on every beat, this many dB against the music"

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
    generate.add_argument(
        "--words", help="draw every section's prompt from this list, by name or path"
    )
    generate.add_argument(
        "--section-every", type=parse_duration,
        help="time between new sections, such as 3.5m (default 3.5m with --words)",
    )
    generate.add_argument(
        "--save-journey", type=Path,
        help="write the sections to a timeline to read or edit, and stop",
    )
    generate.add_argument("--drums", type=float, help=DRUMS_HELP)
    generate.add_argument("--kick", type=float, help=KICK_HELP)
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
    master.add_argument("--drums", type=float, help=DRUMS_HELP)
    master.add_argument("--kick", type=float, help=KICK_HELP)

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


def load_words(name: str) -> WordList:
    path = Path(name)
    if not path.exists():
        path = WORDLISTS_DIR / f"{name}.txt"
    if not path.exists():
        lists = ", ".join(sorted(found.stem for found in WORDLISTS_DIR.glob("*.txt")))
        raise SystemExit(f"no word list called {name!r}; the lists here are: {lists}")
    try:
        return parse_wordlist(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise SystemExit(f"{path}: {error}") from error


def choose_evolution(args: argparse.Namespace) -> Evolution | None:
    evolution = EVOLUTIONS[args.evolve]
    if args.section_every and (args.journey or evolution is None):
        raise SystemExit("--section-every can't be used with --journey or --evolve off")
    section_s = args.section_every or (WORD_SECTION_S if args.words else None)
    return replace(evolution, section_s=section_s) if evolution and section_s else evolution


def save_journey(plan: Plan, path: Path) -> int:
    if path.exists():
        raise SystemExit(f"already exists: {path}")
    command = f"uv run longdrive generate --journey {path} --duration {plan.duration_s:g}"
    if plan.drums_db is not None:
        command += f" --drums {plan.drums_db:g}"
    if plan.kick_db is not None:
        command += f" --kick {plan.kick_db:g}"
    path.write_text(
        f"# {plan.movements} sections at {plan.bpm} BPM, from seed {plan.seed}. Edit any line,"
        f" then render it:\n#\n#   {command}\n\n{journey_text(plan)}",
        encoding="utf-8",
    )
    print(f"Wrote {plan.movements} sections to {path}\nRender it with: {command}")
    return 0


def command_generate(args: argparse.Namespace) -> int:
    if not args.prompt and not args.journey and not args.words:
        raise SystemExit("give a prompt, a word list with --words, or a timeline with --journey")
    if args.words and (args.prompt or args.journey):
        raise SystemExit("--words writes the prompts, so give it no prompt or --journey")
    journey = parse_journey(args.journey.read_text(encoding="utf-8")) if args.journey else None
    words = load_words(args.words) if args.words else None
    if words and not (args.bpm or words.bpm):
        raise SystemExit(f"{args.words} has no tempo: add a 'bpm:' line to it, or use --bpm")
    plan = make_plan(
        args.prompt,
        args.duration,
        seed=args.seed,
        bpm=args.bpm,
        key=args.key,
        evolution=choose_evolution(args),
        journey=journey,
        words=words,
        drums_db=args.drums,
        kick_db=args.kick,
    )
    if journey and plan.movements < len(journey):
        print(
            f"Dropped {len(journey) - plan.movements} sections too short to play outside "
            "their blends.",
            file=sys.stderr,
        )
    if args.save_journey:
        return save_journey(plan, args.save_journey)
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
    if not run.committed_paths():
        raise SystemExit("no finished segments yet")
    if args.drums is not None:
        run.plan = replace(run.plan, drums_db=args.drums)
    if args.kick is not None:
        run.plan = replace(run.plan, kick_db=args.kick)
    run.work_dir.mkdir(exist_ok=True)
    for path in finish(run, tuple(args.format)):
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
