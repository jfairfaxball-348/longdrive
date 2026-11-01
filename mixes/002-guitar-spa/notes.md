# 002: Guitar Spa

A full hour of relaxing guitar-led background music at 72 BPM, in 17 sections with a new prompt every
3:33, and a very soft pulse in place of the model's drums.

| | |
| --- | --- |
| Master | `longdrive-002-guitar-spa.wav`, marked read-only |
| Format | 24-bit, 48 kHz stereo WAV, 1.04 GB |
| Length | 1:00:00 (172,800,000 frames) |
| SHA-256 | `a162b62885c9fc52489bee064a0c0602bf15af9caa31e4ac2b12147761e3819c` |
| Finished | 2026-11-01, after 3 hours 44 minutes of rendering |
| Code | Git tag `mix-002` |

The two MP3s are 128 kbps copies for a phone, split at 30:00.

## Sections

| Starts | Key asked for | Section |
| --- | --- | --- |
| 0:00 | G major | New age guitar, stripped back: baritone, hollow-body and twelve-string |
| 3:33 | C major | Spa music: nylon and Spanish guitar, harp, flute, water and waves |
| 7:07 | F major | Zen ambient: harmonics and slide guitar, synth pads and rain |
| 10:40 | C major | Ambient guitar: clean electric with delay, cello and felt piano |
| 14:13 | G major | Wellness music: bowed electric and swells, singing bowls and birdsong |
| 17:47 | E minor | Acoustic ambient: nylon melody and slide guitar, string pads and glass bells |
| 21:20 | B minor | Neoclassical guitar: arpeggios, cello, flute, ocean and rain |
| 24:53 | E minor | Meditative ambient: nylon and clean electric guitar, pads and a stream |
| 28:27 | G major | Wellness music: bowed and Spanish guitar, sub drone and birdsong |
| 32:00 | E minor | Meditative ambient: nylon melody, singing bowls, harp, ocean and birdsong |
| 35:33 | A minor | New age guitar: steel-string and hollow-body, cello drone and glass bells |
| 39:07 | E minor | Zen ambient: hollow-body and bowed guitar, cello, felt piano and rain |
| 42:40 | G major | Spa music: nylon melody and baritone guitar, sub drone and synth pads |
| 46:13 | D major | Acoustic ambient: Spanish guitar and clean electric, glass bells, rain and a stream |
| 49:47 | B minor | Ambient guitar: swells and harmonics, flute and string pads |
| 53:20 | D major | Neoclassical guitar: twelve-string and slide, cello, singing bowls and birdsong |
| 56:53 | G major | New age guitar, the opening section again |

## How it was made

```bash
uv run longdrive generate --words guitar-spa --seed 26 --evolve strong -q standard --name guitar-spa-hour
```

- **Model:** ACE-Step 1.5 `acestep-v15-turbo` at 8 steps, on engine commit `ca1e85f` and weights
  revision `19671f4`.
- **Quality:** `standard`, with three renders per window: 206 renders for 68 windows, after two
  breakdowns were replaced.
- **Drums:** Demucs 4.1.0 (`htdemucs`) took the model's drums down to -60 dB after rendering, and a
  plain kick went in at -13 dB.
- **Machine:** RTX 3050 Ti laptop GPU with 4 GB of VRAM, Python 3.12.10, PyTorch 2.7.1 with CUDA 12.8.

| File | What it records |
| --- | --- |
| `journey.txt` | The timeline, with every prompt in full |
| `plan.json` | The prompt, key and seed planned for each window |
| `state.json` | The seed chosen for each window, and how it measured |
| `engine-packages.txt` | The engine's Python packages and their versions |

The chosen windows are in `runs/guitar-spa-hour/segments/`, at 1.7 GB, with the drum stem in `work/`
beside them, at 1.3 GB. `longdrive master` needs the windows to remaster the hour, at a new kick level
for example, so that run should be kept. Rendering the timeline again with the same seed has not been
tested, so don't assume it comes out identical.

Measurements are in [the engineering notes](../../docs/engineering-notes.md#the-spa-hour).
