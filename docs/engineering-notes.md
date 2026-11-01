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
same tempo and on the same grid. So every journey line is a whole prompt, and a keyword list draws a
whole new one for each section.

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

## Keyword lists

A list of about 50 phrases, grouped by role and drawn fresh for each section, was meant to do what
rewriting prompts by hand did. It does. Six draws from `funky` and `ambient`, rendered once each for
140 s, compared with the sections of the first hour:

| Measure | Mix 001's sections | Six draws |
| --- | --- | --- |
| Busyness | 3.3 to 5.8 | 2.8 to 5.0 |
| Tone between any two | 0.9 to 2.0 dB | 1.0 to 4.3 dB |

These are single renders, not a finished hour. Tone moved with the draw, not with small changes to
a prompt: the same draw and seed, with and without a drum sentence, stayed within 0.5 to 1.2 dB.

Spacing needed one fix. A section used to start at the first window at or after its time, so `3:30`
missed a boundary by half a second and slipped to 4:22. It now starts on the nearest window, and one
too short to play for a phrase outside the blend into it is dropped.

## Busy drums

Mix 001's drums are busy, and take up a lot of the mix. `scripts/drums.py` folds a passage's onsets
into one bar and measures it. Beat share is how much of the low end falls on the four beats: 0.88
for a synthetic four-on-the-floor and 0.39 for a broken kick. Mix 001 averages 0.40, and the only
section that looks like a plain four-on-the-floor is the atmospheric one at 42:46. The top end is hit
on nearly every 16th.

The model could not be talked out of this. Each row is the same prompts and seeds, on `turbo`:

| Attempt | Beat share | Mids against the rest |
| --- | --- | --- |
| No drum sentence | 0.46 | -11.5 dB |
| "Simple four-on-the-floor kick drum." in the frame | 0.51 | -12.4 dB |
| A similar sentence at the very start of the prompt | 0.49 | -13.7 dB |
| A plain kick loop as context, thrown away afterwards | 0.41 | -14.7 dB |
| The same loop with hats | 0.47 | -14.9 dB |
| Cover mode from kick and hats, strength 0.8 | 0.76 | -21.6 dB |
| Cover mode from a kick alone, strength 0.65 | 0.82 | -20.8 dB |
| Cover mode from a kick alone, strength 0.8 | 0.87 | -20.7 dB |
| Cover mode from a soft kick under a chord bed, strength 0.65 | 0.58 | -16.4 dB |

Mids are everything from 150 Hz to 4 kHz, where the chords and keys are. The first five rows are six
prompts, the cover rows four, and the last row two.

Words do little, as with any sentence inside a long prompt. On one seed the pattern barely changes
between wordings, and different seeds only reached about 0.5, so rendering more candidates would not
fix it either. Naming the drums even costs room, with the mids falling further the more prominently
a sentence asks for them. So the default frame doesn't mention drums, and `funky` and `ambient` have
no drums group.

Cover mode (`task_type="cover"` with a source audio file, available on `turbo`) is the only thing
that really changed the pattern, and at 0.8 a kick-only source gave a textbook `@   @   @   @   `. But
the source's balance comes straight through. Hats in the source made the top end denser and pulled
renders together, to within 1.5 dB of each other. A kick alone left the mids 6 to 11 dB lower, and
that was the user's complaint: plainer, but taking up even more room. A chord bed under the kick
calmed the top end, with 6 to 8 hits a bar instead of 10 to 15, but the kick stopped being plain.

Cover mode also only reaches one window. Chaining on from it with `repaint` gave a beat share of
0.78, 0.55, 0.63, 0.44, and the next section did the same: 0.77, 0.40, 0.44. Mids did recover, from
-21 to about -14 dB, so later windows are ordinary renders again.

A whole section can be a single render, which would avoid that: a 300 s window fitted in 4 GB and
took 101 s. The kick still wandered in that one test, and a long render has no selection or context,
so this has not been taken further.

What works is to leave the render alone and fix the mix. Demucs (`htdemucs`) splits out the drums,
which takes 20 s for 140 s of audio on this GPU, in 4 GB. Subtracting that stem from the original
leaves everything else exactly as rendered, where rebuilding from stems would not:

| Passage | Mids | Busyness |
| --- | --- | --- |
| Micro-funk, as rendered | -11.7 dB | 4.14 |
| The same, drums out, soft kick at -9 dB | -8.3 dB | 2.75 |
| Lo-fi deep house, as rendered | -13.8 dB | 4.26 |
| The same, drums out, soft kick at -9 dB | -8.8 dB | 2.32 |

That busyness is close to the 2.54 of the dub techno rewrite, the calmest in these notes. Beat share
hardly moves, at 0.37 to 0.45, because the model's bassline shares that band and is much louder than
a soft kick. The low end of the groove is still the bassline's, so a very plain pulse needs plainer
bass entries in the list as well.

Before settling on Demucs, median-filter separation of sustained and sharp sounds was tried, since it
needs no new dependency. It lowered busyness from 4.7 to 3.2 but made the mids worse, from -18 to
-20 dB. A kick's weight is low and reads as sustained, so it stayed, and sharp parts of the music
went instead.

The added kick needs a grid. Each section's comes from its own drum stem, measured past the blend.
Sections can disagree, and jumping between grids made gaps of 0.85 and 1.5 beats in a test piece,
which would be a stumble in the pulse. So the kick eases across each blend, where even half a beat
is a tempo change of about 0.5%. A section with no drums to measure carries the last grid on.

## Slow music with no drums

The `guitar-spa` list asks for no drums at 72 BPM, on `turbo`. A 14-minute draft in four sections
took 17 minutes, with nothing flagged:

| Measure | The first hour | Guitar spa, in one key |
| --- | --- | --- |
| Busyness, by section | 3.3 to 5.8 | 1.5 to 1.9 |
| Tone between sections | 0.9 to 2.0 dB | 2.0 to 2.3 dB |
| Harmonic similarity | 0.91 to 1.00 | 0.84 to 1.00 |

Tone and harmony moved further than in the first hour, even with the key asked for staying the same.
The model did as it was told for seven minutes, with a drum stem at -54 and -46 dB, then added drums
anyway, at -22 and -24 dB. This is the old finding that calm music gains drums, and taking the drum
stem out after rendering deals with it, so `sft` wasn't needed here.

There are no kicks for a new section to line up with, so sections aren't beat-matched to each other,
and tempo reads are less reliable. One 0.6 s dip to -51 dB at 9:15 hasn't been judged by ear.

## The spa hour

`guitar-spa` rendered for a full hour on `turbo` at standard quality in 3 hours 44 minutes: 68 windows
in 17 sections and 206 renders, with 2 breakdowns rejected and replaced, none flagged, 3.46 GB peak
VRAM and 11.5 GB of worker RAM. Taking the drums out and mastering took 4 minutes more. It uses seed
26 with `--evolve strong`, picked from a scan of seeds for a key path that stays within two steps of
G major and closes gently, which gave 7 keys.

| Measure | The first hour | The spa hour |
| --- | --- | --- |
| Busyness, by section | 3.3 to 5.8 | 1.5 to 2.0 |
| Tone between sections | 0.9 to 2.0 dB | 1.2 to 4.6 dB |
| Tone against the start, by the minute | Up to 3.1 dB | Up to 6.2 dB |
| Harmonic similarity | 0.91 to 1.00 | 0.68 to 0.99 |

The words did what they were there for. Keys carried change as well this time: the key asked for was
one of the top two estimates in 14 of 17 sections, and the three misses were minor keys read as a
close neighbour. These estimates are rough, as before.

Twice as many sections did not thin the music out. First windows averaged a busyness of 1.49 against
1.70 for the rest, about the same gap as mix 001, and the two halves of the hour match at 1.64 and
1.65. Over the whole hour, mids sit at -4.5 dB against the rest, with a beat share of 0.43.

"No drums, no percussion" only went so far. The drum stem stayed under -40 dB in 7 of the 17 sections,
and reached -21.5 dB at 14:13 and -24.6 dB at 56:53, all of it swapped out. Those 7 sections have no
beat of their own to measure, so the kick carries its grid through them. It held across 4,321 hits:
the only gaps more than 5 ms off the beat are 64 that fall 6 ms short, all in the one blend at 28:27.

The tempo read 72 in 12 sections, 71 in one and about 70 in the other four, but this is a low-end
read of slow music, so it may mean little. Two things stand out, and the user asked not to chase them
before hearing the piece: a 0.5 s gap down to -48.8 dB at 56:30, and a tone jump of 11.3 dB at 40:00,
nearly twice mix 001's largest. The user then approved the hour as mix 002, without asking for either
to be changed.

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
