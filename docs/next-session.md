# Next session: word-list prompts and 3.5-minute sections

To pick this up, open a new session in this repo and paste everything below the line, or just say
"read docs/next-session.md and follow it".

---

You're picking up longdrive, which turns prompts into continuous long-form music on this laptop.
Mix 001, a one-hour piece in nine sections, is finished and the user likes it. Measured, it changes
in arrangement and pattern but much less in tone and key, and the user wants the next mixes to
change more often.

Build the two features below. They belong together: a section every 3.5 minutes means 17 prompts an
hour, which nobody wants to write by hand. Live streaming also came up, but the user has parked it,
so leave it alone.

Read these first, and don't repeat the experiments they describe:

- `README.md`, for how longdrive is used.
- `docs/engineering-notes.md`, especially "Making a long piece evolve" and "The first full hour".
- `src/longdrive/plan.py` and `src/longdrive/cli.py`, where most of this work belongs.
- `examples/deep-house-journey.txt`, for prompts that are known to work.
- `mixes/README.md`, for how a finished mix is kept.

## 1. Word-list prompts

The user's request: keep source lists of about 50 words, such as an "ambient" list or a "funky"
list. Choose one list for a piece, then randomly pick about 10 words at a time to make each prompt.

This is the most promising way to get more change. Rewriting a prompt is the only thing that has
really moved the music, taking busyness from 6.9 down to 2.9–4.5. `--evolve`, which reshuffles the
user's own phrases, barely moved it, staying at 5.6 to 6.0. A big pool of new words should do what
hand-rewriting did, but that's untested, so prove it first.

Start from this design, and change whatever doesn't hold up:

- Keep lists in `wordlists/`, one entry per line. Entries can be short phrases, such as "warm
  syncopated synth bass". Write `funky.txt` and `ambient.txt` as starters, with about 50 entries
  each. Use the wording in the example journey, and the ACE-Step guide's advice to name genre,
  instruments, mood and production style (`engines/ace-step/docs/en/ace_step_musicians_guide.md`).
- Group each list by role under headings, such as genre, drums, bass, keys, texture and mood, and
  draw a set number from each. **Ask the user about this before building it.** They asked for flat
  lists, and I recommended groups because a flat draw can land on ten moods and no drums or bass.
  They haven't decided. A list with no headings should work as one group either way.
- Give each section its own draw from the plan's seeded `rng`, so the same seed gives the same
  prompts. Use shuffled batches, as `automatic_journey` does with `DIRECTIONS`, so that every entry
  gets used before any repeats.
- Put a fixed frame around the draw, such as
  `Instrumental {genre}, {bpm} BPM. {entries}. Smooth uninterrupted groove.` All of mix 001's
  prompts open that way, and most close that way.
- Add something like `--words wordlists/funky.txt` to `generate`. The prompts must end up in
  `plan.json` like any other section, so that `--dry-run`, `resume` and `status` keep working, and
  the user can read every prompt before a long render.

Things to watch for:

- **Tempo:** with no prompt text to read a tempo from, `make_plan` picks 40 to 50 BPM. Word-list runs
  need `--bpm` or a tempo in the list, and every section must share it, because the blends depend on
  beat-matching.
- **Required prompt:** `command_generate` exits without a prompt or `--journey`, and `make_plan`
  raises on an empty prompt.
- **Short prompts:** mix 001's prompts run to about 35 words. Nobody has tested whether 10 words
  hold a steady groove.
- **Ambient needs more than words:** the notes show sparse music holds up better with `--model sft`,
  which takes 2.5 times as long, and with slow tempos, while `turbo` drifts busier. `sft` has also
  never been run through a journey. Test an ambient list separately, and tell the user what you find.
- **Words that pull against a groove:** the model treats everything as a song, so leave out words
  like "breakdown", "drop", "intro", "outro", "solo" and "vocals".
- **Taste:** the quality checks judge steadiness, not whether a prompt is any good, so a bad draw
  will still render for hours.
- **Keys:** these are a nudge at best. Mix 001 mostly sat around E minor whatever it asked for, so
  don't spend effort here.

Not requested, but worth offering once the basics work: sliding from one list to another across the
hour, for example from funky to ambient.

## 2. A new section every 3.5 minutes

The user's request: a regular new section throughout the piece, about every 3.5 minutes. They know
this means more change, and that's what they want.

A timeline can nearly do this today. Lines at multiples of 3:29 give one section every 96 bars
(209.45 s) at 110 BPM, which is 17 sections in an hour. What's missing:

- **A way to ask for it:** add something like `--section-every 3.5m`, which works with `--words` and
  with `--evolve`. `parse_duration` already reads `3.5m`, and `Evolution.section_s` is the existing
  setting. Make the spacing a whole number of windows at any tempo.
- **A rounding fix:** `section_at` starts each section at the first window boundary at or after its
  time. A window is 52.36 s at 110 BPM, so `3:30` misses a boundary by half a second and slips to
  4:22, and mix 001's `7:00` landed at 7:51. Round to the nearest window instead. Saved plans aren't
  affected, because `resume` reads `plan.json`.
- **A minimum length:** drop any section too short to play outside its blends. At 3:29 spacing, an
  eighteenth section would start at 59:21, inside the closing fade. `automatic_journey` already
  counts 17, but a hand-written timeline doesn't stop you.

Things to watch for:

- **Blends:** these stay at 24 bars out of 96, so a quarter of the piece is blending, and each
  section plays alone for about 2:37. The user has been told this. If it feels restless, try 16 bars
  with `blend_s=32`. `Timing` has no command-line option yet.
- **Energy sag:** a new section has no character to match yet, so selection picks the calmest of its
  renders. In mix 001, first windows averaged a busyness of 4.1, against 4.6 for the three after
  them. Twice as many sections could pull the whole piece sparser.

Render time doesn't change, because an hour is still 69 windows: about 4 hours 15 minutes at
`standard`.

## Testing it

Add tests in the style of `tests/test_plan.py` and `tests/test_cli.py`. `uv run pytest` runs on the
synth engine, so it needs no GPU, and takes about 3.5 minutes. Also run
`uv run ruff check src tests scripts`.

Then work up with the real model, stopping if a step fails:

1. **Single renders:** draw about six prompts and render each once for 140 s, using `AceStepEngine`
   and `RenderRequest` directly, at about 75 s per render. Compare them with `audio.busyness` and
   `audio.spectral_shape`, as the table in the notes does.
2. **A short audition:** render 14 minutes at `--quality draft`, which gives 4 sections and takes
   about 20 minutes. Run `scripts/review_run.py`, send the user an MP3, and ask what they hear.
3. **A full hour:** only with the user's OK, render an hour at `standard`. Save it as `mixes/002-…`,
   following `mixes/README.md`.

Compare against mix 001 with `scripts/`. To beat: whole sections differed in tone by only 0.9 to
2.0 dB, with a harmonic similarity of 0.91 to 1.00. To hold: tempo locked, no stops, no flagged
windows, and kicks together through the middle of each blend.

## Done means

- One command makes an hour from a word list, with a new section about every 3.5 minutes.
- Its `--dry-run` shows 17 evenly spaced sections with different prompts, and the same seed gives
  the same plan.
- `funky` and `ambient` starter lists exist, and the README explains lists and the new option.
- Tests and lint pass.
- The user has heard an audition and said whether it's right.

## Ground rules

- The user judges by ear, and you can't hear audio. Measurements only show whether something is
  worth their time. Files must be under 30 MiB to reach their phone, which `scripts/phone_mp3.py`
  handles.
- Leave `mixes/001-deep-house-journey/` and `runs/deep-house-journey-1h/` alone. Remastering mix 001
  needs that run's segments.
- Disk space is tight, with about 40 GB free. An hour takes about 2.6 GB. Clear test runs as you go.
- The GPU has 4 GB, so run one render at a time, run long renders in the background, and don't edit
  the worker while a run is live.
- Commit only when the user asks, and branch off `main` first.
