# longdrive

Give it a prompt, walk away, and come back to an hour of continuous instrumental music that holds
its groove and evolves as it goes. It all runs on your own machine, using open models.

```bash
uv run longdrive generate "Instrumental minimal deep house, 110 BPM, warm synth bass" --duration 1h
```

The finished piece appears in `runs/` as a 24-bit, 48 kHz WAV, which is lossless from the model
onwards. An hour comes to about 1 GB. Pieces over about four hours pass the 4 GB limit of a plain
WAV, so they use its extension, RF64, which some older software can't open.

## How it works

No open model can write an hour of music in one go. The best that fit on a consumer GPU manage a few
minutes, and they treat the end of that as the end of a song. So longdrive builds a long piece the
way a DJ builds a set: from movements, each of which holds a steady groove, mixed into one another.

**Within a movement, renders are chained and steered.** Left to continue, the model behaves like a
songwriter: calm music gains drums, and a steady groove falls into breakdowns. So each window keeps
only its settled start:

```text
  previous audio  |  kept in the piece  |      thrown away
<--- context_s -->|<----- hop_s ------->|<----- tail_s ----->
```

1. **Plan.** The prompt becomes a schedule of windows with a fixed tempo and seeds. The plan is saved
   first, so a resumed run reproduces the same music. Joins land on 8-bar phrase boundaries.
2. **Render.** [ACE-Step 1.5](docs/model-survey.md) outpaints each window from the end of the piece
   so far. The context comes back unchanged, so tempo and harmony carry through the join.
3. **Discard the tail.** The model builds and then fades out across a window. Only the start of
   each is kept, well before either happens.
4. **Choose.** Several candidates are rendered per window. Each is compared with the start of the
   movement, and the one that best holds its energy and tone, with no breakdowns and the smallest
   change at the join, is kept and trimmed to the level of the piece.

**Between movements, the music really changes.** Continuing audio can't go anywhere new, so each
movement starts from a fresh render with its own prompt and key. It is lined up with the outgoing
beat and blended in over about 24 bars. The mids and highs fade slowly across the blend, while the
bass swaps quickly at the midpoint, so that two basslines never clash.

5. **Master.** Everything is joined and streamed to disk with a slow level ride and a soft limiter,
   so an hour of audio never has to fit in memory.

[docs/engineering-notes.md](docs/engineering-notes.md) explains why it works this way, with the
measurements behind each decision.

## Setup

You need [uv](https://docs.astral.sh/uv/), git, and an NVIDIA GPU with at least 4 GB of VRAM.

```bash
uv sync
uv run longdrive setup
```

`setup` clones ACE-Step at a pinned commit into `engines/ace-step`, creates its separate Python
environment, and downloads the weights at pinned revisions. Plan for about 18 GB of disk.

## Usage

```bash
uv run longdrive generate "your prompt" --duration 45m
uv run longdrive status
uv run longdrive resume runs/<name>
uv run longdrive master runs/<name>
```

| Option | Meaning |
| --- | --- |
| `-d, --duration` | Length, such as `45m`, `1h30m` or `3600`. Default `1h`. |
| `-e, --evolve` | `off`, `gentle` or `strong`. See below. |
| `--journey` | A timeline of prompts, for full control. See below. |
| `-q, --quality` | `draft`, `standard` or `best`. See below. |
| `-m, --model` | `turbo` for anything with a groove, `sft` for sparse music without drums. |
| `--bpm`, `--key` | Tempo and key. Also read from the prompt; otherwise chosen slow. |
| `--seed` | Repeat a run exactly. |
| `--engine synth` | Test the pipeline in seconds, with no model or GPU. |

`master` works on a run in progress, so you can listen to what exists so far. It also writes other
formats: `longdrive master runs/<name> --format wav mp3`.

## Making it evolve

What changes a piece is the prompt each movement starts from, so there are three ways to set them.

**Write a journey.** This gives you the most control, and is the way to get the most change. Each
line of a text file is a start time, an optional key, and a complete prompt for that movement:

```text
0:00 [A minor] Instrumental minimal deep house, 110 BPM. Stripped back to kick, bass and hi-hats.
21:00 [C major] Instrumental deep house, 110 BPM. Lush warm chord stabs over a rolling bassline.
35:00 [E minor] Instrumental dub techno, 110 BPM. Deep echoing chords, tape delay, submerged kick.
```

```bash
uv run longdrive generate --journey examples/deep-house-journey.txt --duration 1h
```

[examples/deep-house-journey.txt](examples/deep-house-journey.txt) is an hour-long example to edit.

- **Rewrite, don't add.** Each line replaces the prompt. Adding a sentence to a long prompt
  changes almost nothing, while rewriting it changes the whole arrangement.
- **Keep the tempo the same on every line**, so that movements can be beat-matched.
- Keep keys related. The relative major or minor, or a fifth away, blends cleanly.

**Use `--evolve`.** This is a mild alternative for when you only have one prompt. Each new movement
keeps your opening sentence and uses a different selection of your other phrases, or adds a direction
if the prompt is short. `gentle`, the default, starts a new movement every 10 minutes and moves key
every second one, and `strong` does both every 6 minutes. Expect new patterns and a shift of key,
but much the same sound: the words that set the character of the music don't change, so a journey
is the way to get real change.

**Use `--evolve off`** for one steady movement from start to finish.

## Writing prompts

Your prompt is sent exactly as written. It can be up to 1,000 characters.

- Give a tempo, such as `110 BPM`. Without one, longdrive picks something slow.
- Say what should stay constant: "smooth uninterrupted groove, consistent energy". The model will
  otherwise write a song, with sections.
- For sparse music, ask for it directly, for example "sparse, minimal, sustained, no drums, no
  percussion", and use `--model sft`.

## Quality

| Level | Renders per window | An hour of music takes about |
| --- | --- | --- |
| `draft` | 1 | 1 hour 30 minutes |
| `standard` | 3 | 4 hours 15 minutes |
| `best` | 6 | 9 hours |

Times are for `turbo` at 110 BPM on an RTX 3050 Ti with 4 GB of VRAM. More renders give the
selection more to choose from, which is what keeps breakdowns out. `sft` takes about 2.5 times as
long.

## Leaving it overnight

- Every finished window is saved immediately. If anything stops, `resume` carries on from the
  last finished window.
- The model runs in its own process, which is replaced regularly to stop memory climbing. Crashes
  restart it with a new seed, and a render that hangs is killed.
- A render that keeps failing its checks is kept and flagged in `status`, rather than throwing
  away the night.
- Windows is told not to sleep, but **closing the lid can still suspend the machine**. Stay plugged
  in, and set the lid action to "do nothing" under Power Options.

The log in `runs/<name>/run.log` shows each window's level and busyness, so you can see at a glance
whether a piece is holding its character.

## Keeping a mix

`runs/` is working space. Anything worth keeping moves to [mixes/](mixes/README.md), numbered in
order, along with the recipe that made it.

## Tests

```bash
uv run pytest
uv run pytest -m engine
```

The first runs on the synth engine and a stand-in worker, with no GPU. The second renders with
the real model.

## Licences

longdrive drives ACE-Step 1.5, whose code and weights are MIT-licensed and whose
[model card](https://huggingface.co/ACE-Step/Ace-Step1.5) allows commercial use of output.
