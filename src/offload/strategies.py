"""
strategies.py — Ready-to-use FLUX offloading strategies.

Each strategy function:
  1. Loads the transformer and builds the pipeline via a ModelBase descriptor.
  2. Configures the chosen offloading method.
  3. Runs one generation benchmark (encode + denoise timed separately).
  4. Returns a StrategyResult and cleans up GPU memory.

Usage:
    model = FluxModel("black-forest-labs/FLUX.2-klein-4B")

    r1 = run_full_gpu(model, "a cat", steps=4, device="cuda")
    r2 = run_smart_offload(model, "a cat", max_vram_gb=8.0, steps=4)
    r3 = run_smart_encode(model, "a cat", max_vram_gb=8.0, steps=4)

Available strategies
--------------------
  run_full_gpu               All components on GPU (fastest, ~22 GB VRAM needed)
  run_full_cpu               All components on CPU (slowest baseline, no GPU)
  run_diffusers_offload      diffusers enable_model_cpu_offload()
  run_sequential_cpu_offload diffusers enable_sequential_cpu_offload()
  run_group_offload          diffusers enable_group_offload() + async streams
  run_smart_offload          SmartOffloadManager on transformer; text encoder on CPU
  run_smart_encode           SmartOffloadManager on text encoder + transformer sequentially
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch

from ..models.base import ModelBase
from ..utils.vram import cleanup, reset_peak, vram_peak_gb, vram_reserved_gb
from .offload import SmartOffloadManager, model_total_bytes
from .pipeline_utils import fix_cpu_text_encoder, fix_execution_device, run_generation


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyResult:
    """Timing and VRAM measurements from one strategy run."""
    strategy: str
    encode_s: float
    denoise_s: float
    total_s: float
    peak_gb: float
    steps: int

    @property
    def it_per_s(self) -> float:
        return self.steps / self.denoise_s if self.denoise_s > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save(img, out_dir, filename):
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        img.save(path)
        print(f"  → {path}")


def _header(title: str):
    print("\n" + "═" * 64)
    print(f"  {title}")
    print("═" * 64)


def _print_timing(enc_s, den_s, total_s, peak_gb, steps, cap_gb=None):
    cap = f"  (cap={cap_gb:.1f} GB)" if cap_gb is not None else ""
    print(f"  Encode  : {enc_s:.1f} s")
    print(f"  Denoise : {den_s:.1f} s  ({steps} steps, {steps/den_s:.1f} it/s)")
    print(f"  Total   : {total_s:.1f} s  |  Peak VRAM: {peak_gb:.2f} GB{cap}")


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1 — Full GPU
# ─────────────────────────────────────────────────────────────────────────────

def run_full_gpu(model, prompt, steps=4, device="cuda", seed=42,
                 height=1024, width=1024, guidance_scale=1.0, out_dir=None):
    """All pipeline components on GPU simultaneously. Fastest; requires ~22 GB VRAM."""
    _header("Strategy: FULL GPU")
    transformer = model.load_transformer()
    pipe = model.load_pipeline(transformer).to(device)
    torch.cuda.synchronize()
    print(f"  VRAM after load: {vram_reserved_gb():.2f} GB")

    gen_kwargs = {"height": height, "width": width, "guidance_scale": guidance_scale}
    img, total_s, enc_s, den_s = run_generation(pipe, prompt, steps, gen_kwargs, seed)
    peak = vram_peak_gb()
    _print_timing(enc_s, den_s, total_s, peak, steps)
    _save(img, out_dir, "full_gpu.png")
    del pipe, transformer
    cleanup()
    return StrategyResult("Full GPU", enc_s, den_s, total_s, peak, steps)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2 — Full CPU
# ─────────────────────────────────────────────────────────────────────────────

def run_full_cpu(model, prompt, steps=4, seed=42,
                 height=1024, width=1024, guidance_scale=1.0, out_dir=None):
    """All components on CPU, no GPU. Slowest possible baseline."""
    _header("Strategy: FULL CPU (no GPU)")
    transformer = model.load_transformer()
    pipe = model.load_pipeline(transformer).to("cpu")

    gen = torch.Generator(device="cpu").manual_seed(seed)
    reset_peak()

    t0 = time.perf_counter()
    embeds, _ = pipe.encode_prompt(prompt=prompt, device="cpu")
    enc_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    out = pipe(
        prompt=None,
        prompt_embeds=embeds,
        generator=gen,
        num_inference_steps=steps,
        height=height, width=width, guidance_scale=guidance_scale,
    )
    den_s = time.perf_counter() - t1

    total_s = enc_s + den_s
    _print_timing(enc_s, den_s, total_s, 0.0, steps)
    _save(out.images[0], out_dir, "full_cpu.png")
    del pipe, transformer
    cleanup()
    return StrategyResult("Full CPU (no GPU)", enc_s, den_s, total_s, 0.0, steps)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3 — diffusers enable_model_cpu_offload()
# ─────────────────────────────────────────────────────────────────────────────

def run_diffusers_offload(model, prompt, steps=4, device="cuda", seed=42,
                          height=1024, width=1024, guidance_scale=1.0, out_dir=None):
    """
    diffusers' built-in accelerate hooks, component-level granularity.
    No VRAM budget control; each whole component occupies GPU during its forward.
    """
    _header("Strategy: diffusers enable_model_cpu_offload()")
    transformer = model.load_transformer()
    pipe = model.load_pipeline(transformer)
    pipe.enable_model_cpu_offload()
    torch.cuda.synchronize()
    print(f"  VRAM after hook setup: {vram_reserved_gb():.2f} GB")

    gen_kwargs = {"height": height, "width": width, "guidance_scale": guidance_scale}
    img, total_s, enc_s, den_s = run_generation(pipe, prompt, steps, gen_kwargs, seed)
    peak = vram_peak_gb()
    _print_timing(enc_s, den_s, total_s, peak, steps)
    _save(img, out_dir, "diffusers_offload.png")
    del pipe, transformer
    cleanup()
    return StrategyResult("diffusers model_cpu_offload", enc_s, den_s, total_s, peak, steps)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3b — diffusers enable_sequential_cpu_offload()
# ─────────────────────────────────────────────────────────────────────────────

def run_sequential_cpu_offload(model, prompt, steps=4, device="cuda", seed=42,
                                height=1024, width=1024, guidance_scale=1.0, out_dir=None):
    """
    diffusers enable_sequential_cpu_offload() — sub-module level granularity.
    Every sub-module of every component is moved to GPU individually.
    Minimum VRAM usage; no budget control; no pinned memory.
    """
    _header("Strategy: diffusers enable_sequential_cpu_offload()")
    transformer = model.load_transformer()
    pipe = model.load_pipeline(transformer)
    pipe.enable_sequential_cpu_offload()
    torch.cuda.synchronize()
    print(f"  VRAM after hook setup: {vram_reserved_gb():.2f} GB")

    gen_kwargs = {"height": height, "width": width, "guidance_scale": guidance_scale}
    img, total_s, enc_s, den_s = run_generation(pipe, prompt, steps, gen_kwargs, seed)
    peak = vram_peak_gb()
    _print_timing(enc_s, den_s, total_s, peak, steps)
    _save(img, out_dir, "sequential_offload.png")
    del pipe, transformer
    cleanup()
    return StrategyResult("Sequential CPU offload", enc_s, den_s, total_s, peak, steps)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3c — diffusers enable_group_offload()
# ─────────────────────────────────────────────────────────────────────────────

def run_group_offload(model, prompt, steps=4, device="cuda", seed=42,
                      height=1024, width=1024, guidance_scale=1.0, out_dir=None):
    """
    diffusers enable_group_offload() — block-level offload with async CUDA streams.

    Applied to the transformer only (not the full pipeline). The VAE is moved
    to GPU and the text encoder is kept on CPU with the appropriate pipeline
    patch. Async stream overlap: H→D transfers of the next block run behind
    the current block's compute. Requires num_blocks_per_group=1 for streams.
    """
    _header("Strategy: enable_group_offload() block_level + streams")
    transformer = model.load_transformer()
    pipe = model.load_pipeline(transformer)
    pipe.transformer.enable_group_offload(
        onload_device=torch.device(device),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        use_stream=True,
        num_blocks_per_group=1,  # streams only supported with 1 block per group
    )
    pipe.vae.to(device)
    fix_cpu_text_encoder(pipe, device)
    torch.cuda.synchronize()
    print(f"  VRAM after hook setup: {vram_reserved_gb():.2f} GB")

    gen_kwargs = {"height": height, "width": width, "guidance_scale": guidance_scale}
    img, total_s, enc_s, den_s = run_generation(pipe, prompt, steps, gen_kwargs, seed)
    peak = vram_peak_gb()
    _print_timing(enc_s, den_s, total_s, peak, steps)
    _save(img, out_dir, "group_offload.png")
    del pipe, transformer
    cleanup()
    return StrategyResult("Group offload block_level+streams", enc_s, den_s, total_s, peak, steps)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 4 — SmartOffloadManager (transformer only)
# ─────────────────────────────────────────────────────────────────────────────

def run_smart_offload(
    model,
    prompt,
    max_vram_gb,
    steps=4,
    device="cuda",
    seed=42,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    out_dir=None,
):
    """
    SmartOffloadManager on the transformer only; text encoder stays on CPU.

    The manager fits as many transformer modules as possible into:
        max_vram_gb − already_on_gpu − inference_headroom
    and streams the rest from pinned RAM via async CUDA transfers.
    The text encoder (Qwen3, ~7.6 GB) runs on CPU; its embeddings are moved
    to GPU before the denoising loop. Encoding is slow on CPU.
    """
    _header(f"Strategy: SmartOffloadManager transformer (cap={max_vram_gb:.1f} GB)")
    transformer = model.load_transformer()
    pipe = model.load_pipeline(transformer)
    pipe.text_encoder.to("cpu")
    fix_cpu_text_encoder(pipe, device)
    pipe.vae.to(device)
    print(f"  text_encoder: CPU  ({model_total_bytes(pipe.text_encoder)//1024**2} MB)")
    print(f"  VAE: GPU  ({model_total_bytes(pipe.vae)//1024**2} MB)")
    print(f"  VRAM after VAE: {vram_reserved_gb():.2f} GB")

    headroom = model.activation_headroom_gb(height, width)
    mgr = SmartOffloadManager(
        pipe.transformer,
        max_vram_gb=max_vram_gb,
        device=device,
        num_streams=2,
        extra_reserved_gb=headroom,
    )
    mgr.load()
    s = mgr.summary()
    print(f"  activation headroom: {headroom:.2f} GB  |  {mgr}")
    print(f"  Weights: {s['resident_mb']} MB resident + {s['streaming_mb']} MB paged")
    print(f"  VRAM after transformer load: {vram_reserved_gb():.2f} GB")

    gen_kwargs = {"height": height, "width": width, "guidance_scale": guidance_scale}
    img, total_s, enc_s, den_s = run_generation(pipe, prompt, steps, gen_kwargs, seed)
    peak = vram_peak_gb()
    _print_timing(enc_s, den_s, total_s, peak, steps, cap_gb=max_vram_gb)

    mgr.unload()
    _save(img, out_dir, f"smart_{max_vram_gb:.0f}gb.png")
    del pipe, transformer
    cleanup()
    return StrategyResult(
        f"Smart offload TR ({max_vram_gb:.0f}GB cap)",
        enc_s, den_s, total_s, peak, steps,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 5 — SmartOffloadManager on text encoder + transformer (sequential)
# ─────────────────────────────────────────────────────────────────────────────

def run_smart_encode(
    model,
    prompt,
    max_vram_gb,
    steps=4,
    device="cuda",
    seed=42,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    out_dir=None,
):
    """
    SmartOffloadManager applied to both the text encoder AND the transformer,
    sequentially, so each phase gets the full max_vram_gb budget independently.

    Phase 1 (encoding):
        te_mgr.load() → encode_prompt() on GPU → te_mgr.unload() → VRAM freed

    Phase 2 (denoising):
        tr_mgr.load() → pipe(prompt_embeds=…) → tr_mgr.unload()

    The Qwen3 encoding step is GPU-accelerated (much faster than the CPU path).
    We only need fix_execution_device (not fix_cpu_text_encoder) since the
    SmartOffloadManager hooks transparently stream TE layers to GPU during the
    forward, keeping weights and inputs always co-located.
    """
    _header(f"Strategy: SmartOffloadManager TE + TR  (cap={max_vram_gb:.1f} GB)")
    transformer = model.load_transformer()
    pipe = model.load_pipeline(transformer)
    pipe.vae.to(device)
    pipe.text_encoder.to("cpu")
    pipe.transformer.to("cpu")
    # _execution_device must return CUDA so input_ids and latents land on GPU.
    # SmartOffloadManager hooks ensure TE/TR weights reach GPU for each forward.
    fix_execution_device(pipe, device)

    exec_dev = pipe._execution_device
    gen = torch.Generator(device=exec_dev).manual_seed(seed)

    # ── Phase 1: encode ───────────────────────────────────────────────────────
    te_mgr = SmartOffloadManager(
        pipe.text_encoder,
        max_vram_gb=max_vram_gb,
        device=device,
        num_streams=2,
        extra_reserved_gb=0.4,  # Qwen3 activations are small
    )
    te_mgr.load()
    s = te_mgr.summary()
    print(f"  [TE]  {s['resident_mb']} MB resident + {s['streaming_mb']} MB paged")
    print(f"  VRAM before encode: {vram_reserved_gb():.2f} GB")

    reset_peak()
    t0 = time.perf_counter()
    with torch.inference_mode():
        prompt_embeds, _ = pipe.encode_prompt(prompt=prompt, device=exec_dev)
    torch.cuda.synchronize()
    enc_s = time.perf_counter() - t0
    enc_peak = vram_peak_gb()
    te_mgr.unload()
    cleanup()
    print(f"  Encode  : {enc_s:.1f} s  (Qwen3 GPU-streamed, peak {enc_peak:.2f} GB)")

    # ── Phase 2: denoise ──────────────────────────────────────────────────────
    headroom = model.activation_headroom_gb(height, width)
    tr_mgr = SmartOffloadManager(
        pipe.transformer,
        max_vram_gb=max_vram_gb,
        device=device,
        num_streams=2,
        extra_reserved_gb=headroom,
    )
    tr_mgr.load()
    s = tr_mgr.summary()
    print(f"  activation headroom: {headroom:.2f} GB")
    print(f"  [TR]  {s['resident_mb']} MB resident + {s['streaming_mb']} MB paged")
    print(f"  VRAM before denoise: {vram_reserved_gb():.2f} GB")

    reset_peak()
    t1 = time.perf_counter()
    out = pipe(
        prompt=None,
        prompt_embeds=prompt_embeds,
        generator=gen,
        num_inference_steps=steps,
        height=height, width=width, guidance_scale=guidance_scale,
    )
    torch.cuda.synchronize()
    den_s = time.perf_counter() - t1
    den_peak = vram_peak_gb()
    tr_mgr.unload()

    total_s = enc_s + den_s
    print(f"  Denoise : {den_s:.1f} s  ({steps} steps, {steps/den_s:.1f} it/s)")
    print(f"  Total   : {total_s:.1f} s")
    print(f"  Peak enc/den: {enc_peak:.2f}/{den_peak:.2f} GB  (cap={max_vram_gb:.1f} GB)")

    _save(out.images[0], out_dir, f"smart_encode_{max_vram_gb:.0f}gb.png")
    del pipe, transformer
    cleanup()
    return StrategyResult(
        f"Smart offload TE+TR ({max_vram_gb:.0f}GB cap)",
        enc_s, den_s, total_s, max(enc_peak, den_peak), steps,
    )
