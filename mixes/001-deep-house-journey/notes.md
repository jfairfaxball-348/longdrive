# 001: Deep House Journey

The first full hour from longdrive: 110 BPM minimal deep house in nine sections.

| | |
| --- | --- |
| Master | `longdrive-001-deep-house-journey.wav`, marked read-only |
| Format | 24-bit, 48 kHz stereo WAV, 1.04 GB |
| Length | 1:00:00 (172,800,031 frames) |
| SHA-256 | `502cff0db745b6266cce612120498e9cc6b93d0505e872830950fea05c1cf386` |
| Finished | 2026-11-01, after 4 hours 14 minutes of rendering |
| Code | Git tag `mix-001` |

The two MP3s are 128 kbps copies for a phone, split at 30:00.

## Sections

| Starts | Key asked for | Section |
| --- | --- | --- |
| 0:00 | A minor | Stripped-back minimal deep house |
| 7:51 | A minor | Micro-funk |
| 14:50 | A minor | Chord stabs and filter movement |
| 21:49 | C major | Lush chords and electric piano |
| 28:48 | C major | Melodic minimal house |
| 35:47 | E minor | Dub techno |
| 42:46 | E minor | Atmospheric deep house |
| 49:45 | A minor | Layered electronic groove |
| 55:51 | A minor | Stripped back again |

## How it was made

```bash
uv run longdrive generate --journey mixes/001-deep-house-journey/journey.txt -d 1h --seed 110
```

- **Model:** ACE-Step 1.5 `acestep-v15-turbo` at 8 steps, on engine commit `ca1e85f` and weights
  revision `19671f4`.
- **Quality:** `standard`, with three renders per window: 213 renders for 69 windows.
- **Machine:** RTX 3050 Ti laptop GPU with 4 GB of VRAM, Python 3.12.10, PyTorch 2.7.1 with CUDA 12.8.

| File | What it records |
| --- | --- |
| `journey.txt` | The timeline, with every prompt in full |
| `plan.json` | The prompt, key and seed planned for each window |
| `state.json` | The seed chosen for each window, and how it measured |
| `engine-packages.txt` | The engine's Python packages and their versions |

The chosen windows are in `runs/deep-house-journey-1h/segments/`, at 1.6 GB. `longdrive master` needs
them to remaster the hour, so that run should be kept. Rendering the timeline again with the same
seed has not been tested, so don't assume it comes out identical.

Measurements are in [the engineering notes](../../docs/engineering-notes.md#the-first-full-hour).
