"""Render a set of prompts once each and compare them, to test wording before a long run.

The input is a JSON list of {"label", "prompt", "seed", "key"}, with an optional "context" audio
file for the model to continue from. Audio and a results file go in the output folder. A render
that already exists is measured and not repeated.

    uv run python scripts/try_prompts.py prompts.json out_dir [bpm]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from drums import drum_stats, line, pictures

from longdrive import audio
from longdrive.engines import RenderRequest, Settings
from longdrive.engines.ace_step import AceStepEngine

RENDER_S = 140.0
TIMEOUT_S = 900
# What a new section keeps at 110 BPM, leaving out the settling at the very start.
FROM_S = 12.0
TO_S = 104.0


def context_length(item):
    return sf.info(item["context"]).duration if item.get("context") else 0.0


items = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out_dir = Path(sys.argv[2])
bpm = int(sys.argv[3]) if len(sys.argv) > 3 else 110
out_dir.mkdir(parents=True, exist_ok=True)

engine = AceStepEngine(Settings(), log_path=out_dir / "engine.log")
engine.start()
try:
    for item in items:
        path = out_dir / f"{item['label']}.wav"
        if path.exists():
            continue
        began = time.monotonic()
        context = Path(item["context"]) if item.get("context") else None
        engine.render(
            RenderRequest(
                out_path=path,
                caption=item["prompt"],
                seed=item["seed"],
                new_s=RENDER_S,
                bpm=bpm,
                key=item["key"],
                time_signature="4",
                context_path=context,
                context_s=context_length(item),
            ),
            timeout_s=TIMEOUT_S,
        )
        print(f"rendered {item['label']} in {time.monotonic() - began:.0f} s", flush=True)
finally:
    engine.stop()

results = []
for item in items:
    skip = context_length(item)
    x, rate = audio.read_region(out_dir / f"{item['label']}.wav", skip + FROM_S, skip + TO_S)
    stats = drum_stats(x, rate, bpm)
    low, top = pictures(x, rate, bpm)
    shape = audio.spectral_shape(x, rate)
    results.append(
        {**item, **stats, "low": low, "top": top, "shape": [float(band) for band in shape]}
    )
    print(f"{item['label']:18s} |{low}| |{top}| {line(stats)}", flush=True)

shapes = np.array([result["shape"] for result in results])
print("--- tone distance between renders (dB)")
for row in shapes:
    print(" ".join(f"{float(np.abs(row - other).mean()):4.1f}" for other in shapes))
(out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
