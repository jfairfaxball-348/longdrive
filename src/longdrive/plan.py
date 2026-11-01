"""Turn a prompt into a reproducible schedule of overlapping renders.

A piece is rendered as a chain of windows. Each window continues the audio before
it, adds ``hop_s`` of new material, and over-renders by ``tail_s`` so that only the
settled start of each window is kept. The model builds and fades as a window goes on:

    previous audio   |   committed to the piece   |      thrown away
    <-- context_s -->|<--------- hop_s ---------->|<------ tail_s ------>

A piece that evolves is built like a DJ set, from movements. Each starts from a fresh
render with its own prompt and key, chains into a steady groove, and is mixed into the
next over a long beat-matched blend. Fresh renders matter because a window that
continues existing audio cannot change much. The prompts have to really differ too:
adding a sentence to a long prompt changes almost nothing, while rewriting it changes
the whole arrangement.

The plan is written to disk before any rendering starts, so resuming a run
reproduces the same music even if the code or its defaults change later.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

DEFAULT_KEYS = (
    "C major",
    "D major",
    "F major",
    "G major",
    "A minor",
    "C minor",
    "D minor",
    "E minor",
)

# For prompts too short to vary from their own words, where a sentence carries real
# weight. They name changes rather than instruments, so they suit any genre.
DIRECTIONS = (
    "Stripped back, with fewer elements and more space.",
    "A new melodic motif slowly emerges and comes forward.",
    "Fuller and more layered, with richer harmony.",
    "Brighter and more open, with the filter slowly rising.",
    "Darker and deeper, with the weight in the low end.",
    "The rhythm is more intricate, with subtle new details.",
    "A warm sustained layer rises underneath everything.",
    "Sparser and more hypnotic, with elements falling away.",
)

# The prompt, its template and the tempo and key share a 256 token limit, and anything
# past it is cut from the end, which would lose the tempo and key first. A prompt this
# long comes to about 190 tokens, leaving room for the rest.
MAX_PROMPT_CHARS = 1000

PHRASE_BARS = 8

# A prompt needs this many phrases beyond its opening sentence to vary from its own words.
_MIN_PHRASES = 6
_PHRASES_KEPT = 0.4

_NOTES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
_FLATS = {"Db": "C#", "D#": "Eb", "Gb": "F#", "G#": "Ab", "A#": "Bb"}

_BPM_RE = re.compile(r"\b(\d{2,3})\s*bpm\b", re.IGNORECASE)
_KEY_RE = re.compile(r"\b([A-G](?:#|b)?)[ -](major|minor|maj|min)\b")
_SECTION_RE = re.compile(r"^(\d+(?::\d+){1,2})\s+(?:\[([^\]]+)\]\s*)?(.*)$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_PHRASE_RE = re.compile(r",|\band\b")


@dataclass(frozen=True)
class Timing:
    """How windows are laid out, in seconds."""

    hop_s: float = 60.0
    context_s: float = 32.0
    tail_s: float = 80.0
    crossfade_s: float = 4.0
    blend_s: float = 48.0

    def __post_init__(self) -> None:
        if self.context_s > self.hop_s:
            raise ValueError("context_s must not exceed hop_s")
        if max(self.crossfade_s, self.blend_s) > self.tail_s:
            raise ValueError("crossfade_s and blend_s must fit inside tail_s")


@dataclass(frozen=True)
class Section:
    """A movement: its prompt, and which key it is in. Blank means carry on as before."""

    start_s: float
    prompt: str = ""
    key: str | None = None


@dataclass(frozen=True)
class Evolution:
    """How far an automatic journey goes: seconds per movement, and how often keys move."""

    section_s: float
    sections_per_key: int


EVOLUTIONS = {
    "off": None,
    "gentle": Evolution(section_s=600.0, sections_per_key=2),
    "strong": Evolution(section_s=360.0, sections_per_key=1),
}


@dataclass(frozen=True)
class Segment:
    index: int
    start_s: float
    new_s: float
    seed: int
    key: str
    prompt: str
    overlap_s: float
    fresh: bool = False


@dataclass(frozen=True)
class Plan:
    prompt: str
    duration_s: float
    bpm: int
    key: str
    time_signature: str
    seed: int
    timing: Timing
    segments: tuple[Segment, ...] = field(default_factory=tuple)

    @property
    def bar_s(self) -> float:
        return 60.0 / self.bpm * int(self.time_signature)

    @property
    def movements(self) -> int:
        return 1 + sum(segment.fresh for segment in self.segments)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> Plan:
        data = json.loads(text)
        data["timing"] = Timing(**data["timing"])
        data["segments"] = tuple(Segment(**s) for s in data["segments"])
        return cls(**data)

    @classmethod
    def load(cls, path: Path) -> Plan:
        return cls.from_json(path.read_text(encoding="utf-8"))


def snap_to_phrases(timing: Timing, bpm: int, time_signature: str) -> Timing:
    """Fit window lengths to the music, so that every join lands on a phrase boundary."""
    bar_s = 60.0 / bpm * int(time_signature)
    phrase_s = PHRASE_BARS * bar_s
    return replace(
        timing,
        hop_s=max(1, round(timing.hop_s / phrase_s)) * phrase_s,
        context_s=max(1, round(timing.context_s / phrase_s)) * phrase_s,
        crossfade_s=max(1, round(timing.crossfade_s / bar_s)) * bar_s,
        blend_s=max(1, round(timing.blend_s / phrase_s)) * phrase_s,
    )


def parse_bpm(prompt: str) -> int | None:
    match = _BPM_RE.search(prompt)
    return int(match.group(1)) if match else None


def parse_key(prompt: str) -> str | None:
    match = _KEY_RE.search(prompt)
    if not match:
        return None
    mode = "major" if match.group(2).startswith("maj") else "minor"
    return f"{match.group(1)} {mode}"


def related_keys(key: str) -> tuple[str, ...]:
    """The keys that sit closest to ``key``: its relative, and a fifth either way."""
    root, mode = key.split()
    index = _NOTES.index(_FLATS.get(root, root))
    if mode == "minor":
        relative = f"{_NOTES[(index + 3) % 12]} major"
    else:
        relative = f"{_NOTES[(index + 9) % 12]} minor"
    return (
        relative,
        f"{_NOTES[(index + 7) % 12]} {mode}",
        f"{_NOTES[(index + 5) % 12]} {mode}",
    )


def parse_journey(text: str) -> list[Section]:
    """Read a timeline, one ``minutes:seconds [optional key] prompt`` per line."""
    sections = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _SECTION_RE.match(line)
        if not match:
            raise ValueError(f"journey line {number}: expected 'MM:SS prompt', got {raw!r}")
        seconds = 0.0
        for part in match.group(1).split(":"):
            seconds = seconds * 60 + int(part)
        sections.append(Section(seconds, match.group(3).strip(), match.group(2)))
    return sorted(sections, key=lambda section: section.start_s)


def _phrases(prompt: str) -> tuple[str, list[str]]:
    """Split a prompt into its opening sentence and the phrases that follow."""
    sentences = _SENTENCE_RE.split(prompt.strip())
    found = []
    for sentence in sentences[1:]:
        for phrase in _PHRASE_RE.split(sentence):
            cleaned = phrase.strip(" .,!?")
            if cleaned:
                found.append(cleaned)
    return sentences[0], found


def vary_prompt(prompt: str, direction: str, rng: random.Random) -> str:
    """A different take on a prompt, in its own words where it has enough of them.

    The opening sentence stays, since that is usually where the genre and tempo are. A
    different selection of what follows then brings different elements to the fore.
    """
    opening, phrases = _phrases(prompt)
    if len(phrases) < _MIN_PHRASES:
        return f"{prompt} {direction}"
    chosen = sorted(rng.sample(range(len(phrases)), round(len(phrases) * _PHRASES_KEPT)))
    return f"{opening} {', '.join(phrases[index] for index in chosen)}."


def automatic_journey(
    prompt: str, duration_s: float, home_key: str, evolution: Evolution, rng: random.Random
) -> list[Section]:
    """Movements that each take a new angle on the prompt, returning home to finish."""
    count = max(1, round(duration_s / evolution.section_s))
    directions: list[str] = []
    while len(directions) < count:
        batch = list(DIRECTIONS)
        rng.shuffle(batch)
        directions.extend(batch)

    sections = [Section(0.0, prompt, home_key)]
    key = home_key
    for index in range(1, count):
        if index % evolution.sections_per_key == 0:
            going_home = index + evolution.sections_per_key >= count
            key = home_key if going_home else rng.choice(related_keys(key))
        varied = vary_prompt(prompt, directions[index], rng)
        sections.append(Section(index * evolution.section_s, varied, key))
    return sections


def make_plan(
    prompt: str,
    duration_s: float,
    *,
    seed: int | None = None,
    bpm: int | None = None,
    key: str | None = None,
    time_signature: str = "4",
    timing: Timing | None = None,
    evolution: Evolution | None = None,
    journey: list[Section] | None = None,
) -> Plan:
    prompt = prompt.strip() or (journey[0].prompt if journey else "")
    if not prompt:
        raise ValueError("prompt must not be empty")
    if duration_s < 10:
        raise ValueError("duration must be at least 10 seconds")

    seed = random.SystemRandom().randrange(2**31) if seed is None else seed
    rng = random.Random(seed)

    bpm = bpm or parse_bpm(prompt) or rng.randrange(40, 51, 2)
    opening_key = journey[0].key if journey else None
    key = key or opening_key or parse_key(prompt) or rng.choice(DEFAULT_KEYS)
    timing = snap_to_phrases(timing or Timing(), bpm, time_signature)

    if journey is None and evolution is not None:
        journey = automatic_journey(prompt, duration_s, key, evolution, rng)
    sections = journey or [Section(0.0, prompt, key)]

    longest = max(len(section.prompt or prompt) for section in sections)
    if longest > MAX_PROMPT_CHARS:
        raise ValueError(f"prompt is over {MAX_PROMPT_CHARS} characters, and would be cut off")

    count = max(1, math.ceil(duration_s / timing.hop_s))

    def section_at(index: int) -> Section:
        started = [s for s in sections if s.start_s <= index * timing.hop_s]
        return started[-1] if started else sections[0]

    segments = []
    for index in range(count):
        section = section_at(index)
        started = [s for s in sections if s.start_s <= index * timing.hop_s]
        prompts = [s.prompt for s in started if s.prompt]
        keys = [s.key for s in started if s.key]
        next_is_fresh = index + 1 < count and section_at(index + 1) is not section
        segments.append(
            Segment(
                index=index,
                start_s=index * timing.hop_s,
                new_s=min(timing.hop_s, duration_s - index * timing.hop_s),
                seed=rng.randrange(2**31),
                key=keys[-1] if keys else key,
                prompt=prompts[-1] if prompts else prompt,
                overlap_s=timing.blend_s if next_is_fresh else timing.crossfade_s,
                fresh=index > 0 and section_at(index - 1) is not section,
            )
        )

    return Plan(
        prompt=prompt,
        duration_s=duration_s,
        bpm=bpm,
        key=key,
        time_signature=time_signature,
        seed=seed,
        timing=timing,
        segments=tuple(segments),
    )
