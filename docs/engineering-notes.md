# Engineering notes

What testing ACE-Step 1.5 on this hardware turned up. None of this is in upstream's docs,
and most of it shaped the design, so read this before changing how windows are rendered.

All measurements are from an RTX 3050 Ti (4 GB) with an i7-11800H and 15.7 GB of RAM, running
ACE-Step commit `ca1e85f` on Windows 11.

## The model thinks everything is a song

A window ending is treated as the song ending. In the first test the audio fell to -60 dB about
6 seconds before the window ended. Each window therefore over-renders, and the end is thrown
away.

More importantly, the music escalates. A chain of two-minute windows starting from "warm ambient
piano and soft pads" went from -18.5 dB RMS to about -10 dB within six minutes, and from sustained
pads to a full arrangement with drums and bass. This is what the model has learned songs do, and
it is the main obstacle to long, slow music. A spectrogram showed it plainly: dark and sparse for
130 seconds, then broadband hits up to 16 kHz.

The change is easy to detect. Measuring tone colour against the opening gave 0 dB for the opening
and 12.6 dB for the windows with drums, with the band above 6 kHz up by 25 dB. The first minute is
an unrepresentative intro, though: the same measure read 6 to 9 dB against the first minute and a
steady 3 to 6 dB against the first three, so the opening is measured over a longer span.

## What holds the music steady

Each row was a chain of renders from the same prompt, tracking level and busyness.

| Lever | Result |
| --- | --- |
| None | Reaches -12 dB by the second two-minute window |
| Keep only the first 60 s of each window | Helps. Drums tended to arrive about 60 s into a window, whatever came before |
| Calm caption, with 60 s windows | Slows the rise: about 8 minutes to reach -13 dB, not 2 |
| `[Intro - ambient]` structure tag | No clear effect |
| Holding the context at a fixed level | No effect. Raw output stayed within about 0.3 dB of the unlocked run with the context up to 6 dB quieter, so loudness does not feed back through the context |
| `sft` with CFG at 7, calm caption, 44 BPM | Holds. Busyness stayed near 1.1 and level near -16 dB across nine minutes, with no upward trend. Turbo reached busyness 2.3 over the same span |

The `sft` model's classifier-free guidance makes the caption count, and it held calm music far
better than `turbo`, which has guidance baked in. And this is not a ratchet: one turbo window fell
to -20.3 dB straight after one at -12.8 dB, so the music can come back down, and seeds vary by
several dB. That spread is what makes selection work: rendering a few candidates and keeping the
closest pulls the piece back towards the opening at every step.

Other things that were tried or considered:

- A slow tempo helps sparse music directly. At 60 BPM chords changed every 4 to 8 seconds, so the
  default when a prompt gives no tempo is 40 to 50 BPM.
- Style references (`reference_audio`) were not tested. Telling the model a longer song length is
  possible, but with only the start of each window kept there is no ending left to avoid.

## Grooves behave differently

The target for this project turned out to be minimal deep house at 110 BPM, not ambient music, and
most of the ambient findings point the wrong way for it. Adding "no drums, quiet, meditative" would
contradict the prompt, so prompts are now sent verbatim.

Comparing the models on the same long prompt over five windows:

| | `turbo` | `sft` |
| --- | --- | --- |
| Level | Steady at about -9 dB, one breakdown | Moves between -9 and -16 dB, one breakdown |
| Stops | Few | Several: full-band gaps at around 70, 100, 120, 140 and 190 s |
| Low-end correlation between old tail and new head | 0.39 to 0.58 | 0.01 to 0.04 |
| Low-end offset at the join | Under 2 ms | Up to 83 ms |

`turbo` holds a groove better, and both sides of a crossfade carry nearly the same bassline, so it
is the default. The opening problem also inverts: with a groove the failure is falling apart, not
building up. A level dip of more than 9 dB over 8 seconds now counts as a breakdown, and busyness is
penalised in both directions. That judgement came from measurements and spectrograms only, so the
choice between models should be made by ear.

Windows also settle into their own texture, even in calm music. A spectrogram of a run with 250 ms
crossfades showed hard edges in tone at the joins, so joins within a movement fade over about two
bars. `turbo` keeps the kicks within 2 ms across a join, which makes that safe.

## Making a long piece evolve

Holding a piece steady works too well. With one prompt and selection pulling every window towards
the opening, tone stayed within about 2 dB of the start and busyness between 5.0 and 5.6 over eight
minutes. Note patterns do wander by themselves: passages over three minutes apart had a similarity
of 0.28, where 1.0 is identical. But the palette never moves, and that is what makes an hour static.

| Attempt | Tone against the start | Passages 3+ minutes apart | Busyness |
| --- | --- | --- | --- |
| Baseline: one prompt, one key | Under 2.1 dB | 0.28 | 5.0 to 5.6 |
| Chained, adding a sentence and changing key | Under 2.5 dB | 0.12 | 4.8 to 6.7 |
| Fresh renders, adding a sentence and changing key | Under 2.3 dB | 0.14 | 4.7 to 6.6 |

What the model is told barely reaches it through 35 seconds of context, and the keys asked for did
not appear. Fresh renders alone did not fix this either, because the long prompt drowned out the
added sentence. Comparing single renders of the 139-token prompt shows why:

| Prompt | Busyness |
| --- | --- |
| The full prompt | 5.99 |
| The full prompt plus one sentence | 5.94 |
| Rewritten: stripped back | 5.97 |
| Rewritten: melodic electronica | 4.62 |
| Rewritten: lush chords | 3.75 |
| Rewritten: dub techno | 2.54 |

Rewriting the prompt is the lever. Spectrograms of those renders look like different records, at the
same tempo and on the same grid. So every journey line is a whole prompt.

A chain through four rewritten prompts confirmed it: busyness moved from 6.9 down to between 2.9 and
4.5, harmonic similarity to the opening fell to 0.91, and tone stepped by 3 dB at a blend.

Automatic mode is much weaker. It keeps the opening sentence and samples the user's other phrases,
and six such variations stayed at busyness 5.6 to 6.0 and within about 1 dB of each other. The words
that set the character did not change. Without something to write new prompts, that is the limit,
which is why the documentation points to journeys.

Keys do work in fresh renders, within limits. Asking for C major gave C major, D minor gave its
relative F major, and G major gave its relative E minor, but others landed on a close neighbour.
Treat a key as a nudge in a direction.

Separate renders can be mixed because tempo is extremely precise: 110.01, 109.99, 110.00, 110.03 and
110.00 BPM from five renders, with one less clear reading of 110.38. Each new movement is shifted by
up to a bar to line up with the outgoing kicks, which came out at 0 ms through most of each blend.

Re-rendering each join with both sides as context should be smoother still, at the cost of a render
per join. It has not been tried.

## The first full hour

`examples/deep-house-journey.txt` rendered on `turbo` at standard quality in 4 hours 14 minutes:
69 windows and 213 renders, with 6 breakdowns rejected and replaced, none flagged, 3.47 GB peak VRAM
and 11.8 GB of worker RAM. The master is 3600.0006 s long and peaks at -1.0 dBFS with nothing clipped.

The groove held. All nine movements read 109.99 BPM, three-minute level medians stayed between
-15.9 and -16.3 dB, and nothing fell 24 dB below the local level for more than 0.3 s. Kicks lined up
within 10 ms in 20 of 24 checks, including the middle of all eight blends. The other four sit at the
ends of a blend, where only one side's low end is in the mix.

The piece changes in arrangement and pattern, but much less in palette and harmony:

| Measure | One prompt, 8 minutes | The hour |
| --- | --- | --- |
| Busyness, by the minute | 5.0 to 5.6 | 2.8 to 6.1 |
| Passages 3+ minutes apart | 0.28 | 0.15 |
| Tone against the start, by the minute | Under 2.1 dB | Up to 3.1 dB |

Whole movements differ in tone by only 0.9 to 2.0 dB and have a harmonic similarity of 0.91 to 1.00.
For scale, drums arriving measured 12.6 dB. A spectrogram shows the same pitches near 440 Hz running
through much of the hour. Most movements read as E minor or a neighbour whatever key was asked for,
and C major was not clearly reached, though this estimate is rough, with correlations of only 0.3
to 0.4. Keys are not carrying the change.

Above 1 kHz, the visible change is the highs dropping out for 10 to 20 seconds and coming back, as
at 47:42 and 48:05. The roughest join is at 36:39, the largest tone jump of the hour at 6.1 dB, where
the highs come back after about a second with almost nothing above 100 Hz. It falls on a phrase
boundary and may read as a deliberate drop. None of this has been judged by ear.

## Continuation really does continue

`repaint` past the end of the source pads it with silence and outpaints into the gap. In
`conservative` mode the context is re-injected at every step and returns bit-identical (maximum
difference 0.0 outside the 50 ms splice). Chords and bar lines carry across the join, and a held
tone near 440 Hz ran unbroken through twelve minutes and every join.

The join measures a level change, but it is mostly the next chord landing on the beat. Across 8
second windows, ordinary bar lines changed by 1.4 dB on average and joins by 3.1 dB. Comparing
shorter windows mostly measures note decay, so short-window jump detection fires on normal
downbeats.

## Gain

Upstream divides each render by its own peak, then splices in the unscaled context. Generated
audio therefore lands at an arbitrary gain relative to what came before. Peaks of 1.09 to 2.06 were
seen. The worker records the divisor and restores the level. Normalisation is off and intermediate
files are 32-bit float.

Level alone is not a reason to reject a render. An early version rejected anything well above the
opening, and most candidates failed: the model's natural level is around -13 dB, and seeds vary by
several dB. That cost up to six renders per window for something a gain change fixes. Candidates
are now trimmed to the opening's level, which is safe because output level does not depend on the
context. Raw loudness still counts a little in the score, since a louder render often has more
going on.

## Memory

Host memory climbs by about 5.4 GB per render, whatever the audio length, on top of a base of
11.4 GB. This exhausted a 49 GB commit limit by the eighth render, and the worker died with an
access violation (`0xC0000005`) while offloading.

The growth is not visible as Python tensors, and simple device transfers do not cause it, with or
without `inference_mode`. Upstream's heap trimming is `malloc_trim`, which does nothing on Windows.
The cause was not pinned down. Keeping the DiT on the GPU avoids some of it but leaves under 0.5 GB
of VRAM, which pushes the VAE decode to the CPU, at minutes per render.

So every render gets a fresh worker, which holds memory near 12 GB. This is also faster for `sft`:
once a worker passes physical RAM the machine pages, and renders slowed from 150 s to 240 s, while a
reload costs about 35 s. For `turbo` it costs time, about 75 s per render instead of 52 s, but it
removes a whole class of overnight failure. The worker exits with `os._exit`, because normal
interpreter shutdown spent more than 15 seconds freeing a model that the OS reclaims anyway.

## What the model is told

In repaint mode, upstream fills in `duration` from the source audio, which here means the model is
told a 170 second window is a 30 second song.

## The weights run code

`AutoModel.from_pretrained(..., trust_remote_code=True)` executes Python from the checkpoint folder.
Upstream's loader overwrites those files with the copies in its own repository, and after loading
their hashes match that repository. The checkout is pinned, as are the weights' revisions, and
workers run offline with `HF_HUB_OFFLINE=1`.
