"""
pipeline_utils.py — FLUX pipeline patches and generation helpers.

The Flux2KleinPipeline makes some device assumptions that break when model
components are split across CPU and GPU. These utilities work around that.
"""

from __future__ import annotations

import time

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

from ..utils.vram import reset_peak


def fix_execution_device(pipe: Flux2KleinPipeline, device: str) -> None:
    """
    Force ``pipe._execution_device`` to always return ``device``.

    When any pipeline component lives on CPU, the pipeline's property
    may return CPU, causing latents to land on CPU → shape/dtype crash.
    We monkey-patch via a per-instance subclass to avoid global side-effects.
    """
    _dev = torch.device(device)
    pipe.__class__ = type(
        "Fixed_" + type(pipe).__name__,
        (type(pipe),),
        {"_execution_device": property(lambda self: _dev)},
    )


def fix_cpu_text_encoder(pipe: Flux2KleinPipeline, device: str) -> None:
    """
    Patch the pipeline so the Qwen3 text encoder runs on CPU while
    everything else (latents, transformer) runs on ``device``.

    Two patches are required:

    1. ``_execution_device`` → always ``device``, so latents land on GPU.

    2. ``_get_qwen3_prompt_embeds`` (staticmethod) — ``encode_prompt``
       calls it with ``device=self._execution_device`` (now CUDA).  Inside,
       ``input_ids.to(device)`` would move tensors to CUDA but the text
       encoder weights are on CPU → crash.  We intercept, override ``device``
       to ``text_encoder.device`` (CPU), run the static method there, then
       move the resulting embeddings to GPU for the transformer.
    """
    _gpu = torch.device(device)
    _orig_cls = type(pipe)
    _orig_qwen = _orig_cls._get_qwen3_prompt_embeds  # staticmethod → plain fn

    def _patched_qwen(self_p, *args, **kwargs):
        te = kwargs.get("text_encoder") or self_p.text_encoder
        kwargs["device"] = te.device          # run on CPU
        result = _orig_qwen(*args, **kwargs)  # static: no self forwarded
        # move embeddings to GPU for the transformer
        if isinstance(result, tuple):
            return tuple(t.to(_gpu) if isinstance(t, torch.Tensor) else t for t in result)
        if isinstance(result, torch.Tensor):
            return result.to(_gpu)
        return result

    pipe.__class__ = type(
        "Fixed_" + _orig_cls.__name__,
        (_orig_cls,),
        {
            "_execution_device": property(lambda self: _gpu),
            "_get_qwen3_prompt_embeds": _patched_qwen,
        },
    )


def run_generation(
    pipe: Flux2KleinPipeline,
    prompt: str,
    steps: int,
    gen_kwargs: dict,
    seed: int = 42,
) -> tuple[Image.Image, float, float, float]:
    """
    Run one generation pass, timing encode and denoise phases separately.

    Returns
    -------
    (image, total_s, encode_s, denoise_s)
    """
    dev = pipe._execution_device
    gen = torch.Generator(device=dev).manual_seed(seed)
    reset_peak()

    t0 = time.perf_counter()
    embeds, _ = pipe.encode_prompt(prompt=prompt, device=dev)
    torch.cuda.synchronize()
    enc_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    out = pipe(
        prompt=None,
        prompt_embeds=embeds,
        generator=gen,
        num_inference_steps=steps,
        **gen_kwargs,
    )
    torch.cuda.synchronize()
    den_s = time.perf_counter() - t1

    return out.images[0], enc_s + den_s, enc_s, den_s
