# Open music generation models: survey and decision

Researched 31 October 2026 for one job: long, steady, instrumental music from a text
prompt, rendered unattended on a laptop with an RTX 3050 Ti (4 GB VRAM), an 8-core
i7-11800H and 15.7 GB of RAM.

**Decision: ACE-Step 1.5.** It is the only current model that combines a commercial-friendly
licence, a real outpainting mode for extending audio, and a documented path to 4 GB of VRAM. The
`turbo` checkpoint is the default because it holds a groove better; `sft` is available for sparse
music. The rest of this page says why, and what would change the decision.

## The landscape

Generation time was treated as free, so models were ruled out only by memory, licence or
capability. "Fits here" means it can run on the machine above.

| Model | Released | Weights licence | Memory needed | Longest clip | Can extend audio | Fits here |
| --- | --- | --- | --- | --- | --- | --- |
| [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) (2B) | Jan 2026 | MIT, outputs usable commercially | under 4 GB VRAM with INT8 and offload | 10 min (6 min on 4 GB) | Yes: repaint outpainting | **Yes** |
| [ACE-Step 1.5 XL](https://huggingface.co/ACE-Step/acestep-v15-xl-sft) (4B) | Apr 2026 | MIT | 12 GB+ with offload, 20 GB+ without | 10 min | Yes | No, not as shipped |
| [Stable Audio 3 Small](https://github.com/Stability-AI/stable-audio-3) | May 2026 | Stability Community (free under $1M revenue), gated | 1.7–2.4 GB VRAM, or CPU | 120 s | Yes: inpainting | Yes, after accepting terms |
| [Stable Audio 3 Medium](https://github.com/Stability-AI/stable-audio-3) | May 2026 | Stability Community, gated | 5–6.5 GB VRAM, flash-attention | 380 s | Yes | No |
| [Magenta RealTime 2](https://github.com/magenta/magenta-realtime) | Jun 2026 | CC BY 4.0, code Apache 2.0 | Apple Silicon for real time; offline CUDA is Linux-only | Unlimited stream | Native: frame by frame | CPU only on Windows |
| [YuE2](https://github.com/multimodal-art-projection/YuE) | Sep 2026 | CC BY-NC 4.0 with creator permission | 24 GB VRAM, Linux | Full song | Song-focused | No |
| [MiniMax Music 3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) | Aug 2026 | Community licence, commercial terms apply | 22–24 GB; 8 GB with layer streaming | 5 min | No | No |
| [LeVo 2](https://github.com/levo-demo/LeVo) | Mar 2026 | Non-commercial, academic use only | 10–28 GB | 4 min 30 s | No | No |
| [HeartMuLa 3B](https://github.com/HeartMuLa/heartlib) | Jan 2026 | Apache 2.0 | about 8 GB of weights in bf16 | 6 min | No | No |
| [DiffRhythm 2](https://github.com/xiaomi-research/diffrhythm2) | Oct 2025 | Apache 2.0 | Benchmarked on RTX 4090 | 3 min 30 s | No | Unverified |
| [MusicGen](https://github.com/facebookresearch/audiocraft) | 2023 | CC BY-NC 4.0 | about 12 GB at fp16 | 30 s, chainable | Sliding window | Older and non-commercial |
| [Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) | 2024 | Stability Community, gated | 5.9 GB+ | 47 s | No | Superseded |

## Quality

There is no reliable public ranking for this use. The benchmarks that exist are about songs:

- The [YuE2 authors' WildSongBench](https://github.com/multimodal-art-projection/YuE) (192
  prompts, published by a competitor in the field) puts open models in this order: YuE2,
  LeVo 2, HeartMuLa, Muse, ACE-Step 1.5, DiffRhythm 2. Its scores lean on lyric accuracy, which
  means nothing for instrumental music.
- [IT-JIM's independent listening test](https://www.it-jim.com/blog/best-open-source-ai-music-generator)
  (May 2026) found LeVo 2 the most natural-sounding, and ACE-Step 1.5 the best base for tooling,
  continuation and fine-tuning. It also reports a **metallic shimmer** in ACE-Step 1.5, on vocals
  and instrumentals alike. The authors suspect low step counts or the tokenizer, but say they
  have not tested that.
- ACE-Step's [own paper](https://arxiv.org/abs/2602.00744) places it between Suno 4.5 and 5. It
  is the developers' own claim.

So the models that rank above ACE-Step for songs need 2 to 6 times this GPU's memory, are not
licensed for commercial use, and offer no way to extend audio.

## Why ACE-Step 1.5

- **It fits, including the higher-quality checkpoint.** Measured here on a 172 s window, with INT8
  weights and CPU offload: 3.6 GB peak VRAM for `turbo` at 8 steps, and 3.3 GB for `sft` at 50 steps
  with classifier-free guidance.
- **Extending audio is a real feature, not a workaround.** `prepare_padding_info` pads the source
  past its end and the sampler outpaints into the gap. In `conservative` mode, the context is
  re-injected at every step, and comes back bit-identical.
- **It is fast enough to spend compute on quality.** A 172 s window takes about 30 s with `turbo`
  and under 2 minutes with `sft`, which leaves hours to spend on rendering several candidates.
- **The licence is clear.** MIT covers code and weights, and the
  [model card](https://huggingface.co/ACE-Step/Ace-Step1.5) says output may be used commercially.
- **It keeps tempo and harmony.** On the first test, bar lines and chords carried straight
  through the join.

Its main weakness for this job is that it wants to write songs, with sections and breakdowns.
Most of longdrive's design exists to hold that back, as
[the engineering notes](engineering-notes.md) explain.

## Rejected alternatives worth revisiting

- **Stable Audio 3 Small.** The closest competitor, and its AudioSparx training data may suit
  generic instrumentals well. It needs a Hugging Face account and accepting licence terms, stops
  at 120 s per clip, and its permissive tier ends at $1M of revenue. A second engine could be added
  behind `longdrive.engines.Engine`; it is untested here.
- **Magenta RealTime 2.** Designed for exactly the "endless stream" case, trained mostly on
  instrumental music, and free of windows and seams. On native Windows it would run on CPU only,
  and its 20 s receptive field means it has no long-range memory.
- **ACE-Step XL.** Upstream's highest quality. It is rated for 12 GB or more, so it would need
  unsupported layer streaming or CPU inference on this machine.

## With a bigger GPU

ACE-Step XL is the straightforward upgrade at 12 GB and above, since it needs no change to how
longdrive works. At 24 GB, YuE2 becomes possible for non-commercial work, but it is a song model
with no extension mode, so it would need a new engine and its seams would be harder to hide.

## Sources

Links above were checked on 31 October 2026. Hardware and licence claims come from each
project's own README, model card or licence file. Figures measured on this laptop are marked
as such; everything else is as published and was not reproduced here.
