# Next session: mix 3 from the user's own prompts

To pick this up, open a new session in this repo and paste everything below the line, or just say
"read docs/next-session.md and follow it".

---

You're picking up longdrive, which turns prompts into continuous long-form music on this laptop.
Mix 001, a one-hour piece in nine sections, is finished and the user likes it. The next mixes should
change more often, with simpler drums that take up less room.

Keyword lists, a new section every 3.5 minutes and a drum swap are merged into `main` and on GitHub.
Mix 002, a full hour of guitar spa music, was made with them, and the user approved it.

Read these first, and don't repeat the experiments they describe:

- `README.md`, especially "Taming the drums" and "Making it evolve".
- `docs/engineering-notes.md`, especially "Keyword lists", "Busy drums", "Slow music with no drums"
  and "The spa hour".
- `src/longdrive/stems.py` and the word list code in `src/longdrive/plan.py`.
- `wordlists/`, which holds `funky`, `ambient` and `guitar-spa`.
- `mixes/README.md`, for how a finished mix is kept.

## Where it stands

- **Keyword lists work.** Six drawn prompts sat up to 4.3 dB apart in tone, where mix 001's sections
  reached 2.0 dB. The lists are grouped by role, with short phrases and an all-new draw each section,
  including the genre. A piece opens stripped back, and its last section reuses that opening.
- **The drums can't be steered.** Words barely reach them, naming them turns them up, and cover mode
  only lasts one window. Don't go back to these.
- **The drum swap is the route.** `--drums` and `--kick` take the model's drums out with Demucs after
  rendering, and add a plain kick. It takes about 4 minutes for an hour of music, and a new level
  takes 11 s per 14 minutes of music with `longdrive master`. The user chose this version, and the
  levels are set in each list: `-60` and `-9` for `funky` and `ambient`, `-60` and `-13` for
  `guitar-spa`, which they called perfect.
- **The guitar spa hour is mix 002**, in `mixes/002-guitar-spa/`. It's 17 sections in 7 keys, 3:33
  apart. It didn't sag: busyness matches across the two halves of the hour. The tag `mix-002` marks
  the code that made it.

## What's left

1. Make mix 3 from the user's own 17 prompts, once they've sent them. Use a journey file that keeps
   their order and wording, with one tempo and a section about every 3:30. They don't want a preview:
   render the full hour, save it as `mixes/003-…` following `mixes/README.md`, update the docs, then
   commit, tag, push and merge. They asked for all of that on 2026-11-01, so don't stop to ask.
2. Render a 14-minute draft from `funky` or `ambient`. Neither has been rendered through a whole run
   since the frame lost its drum sentence and the lists lost their drums group. `runs/audition-funky`
   is that draft, stopped at window 4 of 17, and `uv run longdrive resume runs/audition-funky` carries
   on from there.
3. With the user's OK, render an hour of it at `standard`, about 4 hours 15 minutes, and save it as
   the next mix, following `mixes/README.md`.

Not requested, but worth offering once that's done: sliding from one list to another across the hour.

Things to watch for:

- **The bassline is the groove now.** With the drums out, a soft kick sits well under the model's
  bass, so beat share stays near 0.4. If the pulse feels unclear, the first thing to try is plainer
  bass entries in the list, before a louder kick.
- **Demucs artefacts:** these can't be measured from here, so ask the user to listen for watery sounds
  where the drums were. `--drums -18` leaves a little of the original in, which may hide them.
- **Blends:** these stay at 24 bars out of 96. If 3.5-minute sections feel restless, try 16 bars with
  `blend_s=32`. `Timing` has no command-line option yet.
- **A whole section as one render:** a 300 s window fitted in 4 GB and took 101 s, which might be
  worth a look one day, but it loses context and selection.

## Testing it

`uv run pytest` runs on the synth engine with a stand-in for Demucs, so it needs no GPU, and takes
about 4 minutes. Also run `uv run ruff check src tests scripts`. `scripts/drums.py` measures the
drums of a file or a run, window by window, and `scripts/try_prompts.py` tests wording in single
renders.

Compare against mix 001 with `scripts/`. To beat: whole sections differed in tone by only 0.9 to
2.0 dB, with a harmonic similarity of 0.91 to 1.00. To hold: tempo locked, no stops, no flagged
windows. Kicks in the blends no longer matter once the drums are swapped, but the model's bass and
chords still have to line up.

## Done means

- One command makes an hour from a word list, with a new section about every 3.5 minutes. This works:
  `uv run longdrive generate --words funky`.
- The user has heard an audition of it and said whether it's right.
- The drums suit the user, at levels set in each list.

## Ground rules

- The user judges by ear, and you can't hear audio. Measurements only show whether something is
  worth their time, and their ear overrules them: they still called the drums too busy after the
  plain-kick tests, which measured as a textbook four-on-the-floor. Files must be under 30 MiB to
  reach their phone, which `scripts/phone_mp3.py` handles. Send a finished render straight away: they
  don't want small measured blips chased before they've heard it.
- Leave `mixes/`, `runs/deep-house-journey-1h/` and `runs/guitar-spa-hour/` alone. Remastering a mix
  needs its run's segments.
- Disk space is tight, with about 29 GB free. An hour takes about 4 GB, including its drum stem.
  Clear test runs as you go: `runs/audition-spa/` (0.9 GB) has done its job.
- The GPU has 4 GB, so run one render at a time, run long renders in the background, and don't edit
  the workers while a run is live.
- Commit and push only when the user asks. `.gitignore` hid `src/longdrive/engines/` by mistake until
  the commit at `mix-002`, so that package isn't in git at the tag `mix-001`.
