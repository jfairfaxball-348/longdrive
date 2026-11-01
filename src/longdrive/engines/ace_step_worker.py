"""ACE-Step render worker.

Runs inside the engine's own environment, so it must not import longdrive. Requests
arrive as JSON lines on stdin and replies leave as JSON lines on stdout; anything
else the libraries print is sent to stderr, which the supervisor keeps as a log.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

LYRICS = "[Instrumental]"
OOM_MARKERS = ("out of memory", "outofmemory", "cuda_error_out_of_memory", "not enough memory")


def is_oom(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in OOM_MARKERS)


def load_handler(engine_root: str, model: str):
    from acestep.handler import AceStepHandler

    handler = AceStepHandler()
    # Upstream refuses to start without its language model, then fetches 3.8 GB of
    # weights. This setup never uses it, and setup has already placed what it needs.
    handler._ensure_models_present = lambda **_: None
    # INT8 weights and full CPU offload are what make the 2B checkpoints fit in 4 GB.
    message, ok = handler.initialize_service(
        project_root=engine_root,
        config_path=model,
        device="cuda",
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=True,
        offload_dit_to_cpu=True,
        quantization="int8_weight_only",
    )
    if not ok:
        raise RuntimeError(message)

    # Upstream divides each render by its own peak and only then splices the context
    # back in, unscaled. Remember the divisor so the new audio can be put back at the
    # context's level; otherwise every window lands at an arbitrary gain.
    decode = handler.tiled_decode

    def tracked_decode(*args, **kwargs):
        wavs = decode(*args, **kwargs)
        handler.longdrive_decode_peak = max(1.0, float(wavs.abs().max()))
        return wavs

    handler.tiled_decode = tracked_decode
    return handler


def render(handler, request: dict, scratch: str) -> dict:
    import psutil
    import soundfile as sf
    import torch
    from acestep.constants import TASK_INSTRUCTIONS
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    context_s = request["context_s"]
    total_s = context_s + request["new_s"]
    task = "repaint" if request["context_path"] else "text2music"

    # "conservative" re-injects the context at every diffusion step, so the join to
    # the previous window is outpainted rather than merely conditioned on it.
    params = GenerationParams(
        task_type=task,
        instruction=TASK_INSTRUCTIONS[task],
        caption=request["caption"],
        lyrics=LYRICS,
        instrumental=True,
        bpm=request["bpm"],
        keyscale=request["key"],
        timesignature=request["time_signature"],
        duration=total_s,
        src_audio=request["context_path"],
        repainting_start=context_s,
        repainting_end=total_s,
        repaint_mode="conservative",
        inference_steps=request["steps"],
        guidance_scale=request["guidance"],
        shift=request["shift"],
        seed=request["seed"],
        enable_normalization=False,
        thinking=False,
        use_cot_metas=False,
        use_cot_caption=False,
        use_cot_language=False,
    )
    config = GenerationConfig(
        batch_size=1,
        use_random_seed=False,
        seeds=[request["seed"]],
        audio_format="flac",
    )

    torch.cuda.reset_peak_memory_stats()
    handler.longdrive_decode_peak = 1.0
    result = generate_music(handler, None, params, config, save_dir=scratch)
    if not result.success or not result.audios:
        raise RuntimeError(result.error or result.status_message or "generation failed")

    rate = result.audios[0].get("sample_rate", 48000)
    audio = result.audios[0]["tensor"].float().cpu().numpy().T
    audio[int(round(context_s * rate)) :] *= handler.longdrive_decode_peak

    out_path = Path(request["out_path"])
    temporary = out_path.with_name(f"{out_path.name}.partial")
    sf.write(temporary, audio, rate, format="WAV", subtype="FLOAT")
    os.replace(temporary, out_path)

    for leftover in Path(scratch).glob("*"):
        leftover.unlink(missing_ok=True)

    return {
        "decode_peak": round(handler.longdrive_decode_peak, 3),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "worker_ram_gb": round(psutil.Process().memory_info().private / 2**30, 2),
    }


def main() -> int:
    channel = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)

    def send(**message) -> None:
        channel.write(json.dumps(message) + "\n")
        channel.flush()

    engine_root, model = sys.argv[1], sys.argv[2]
    try:
        handler = load_handler(engine_root, model)
    except BaseException as error:
        traceback.print_exc()
        send(event="failed", error=str(error)[-2000:])
        return 1
    send(event="ready")

    with tempfile.TemporaryDirectory(prefix="longdrive-") as scratch:
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("op") == "quit":
                break
            try:
                send(event="done", **render(handler, request, scratch))
            except Exception as error:
                text = traceback.format_exc()
                sys.stderr.write(text)
                send(event="error", error=str(error)[-2000:], oom=is_oom(text))
    return 0


if __name__ == "__main__":
    status = main()
    sys.stderr.flush()
    # Freeing a loaded model takes longer than loading one, and the OS reclaims it anyway.
    os._exit(status)
