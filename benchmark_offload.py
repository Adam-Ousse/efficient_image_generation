"""
Benchmark different offloading strategies across quantization levels.

For each model config: loads the model, generates one image, unloads, repeats
N_RUNS times.  The full lifecycle (load → generate → cleanup) is monitored
each time so we can average the resource curves.

Outputs (all inside OUTPUT_DIR):
  - {run_id}/run_{i:02d}_timeseries.csv   raw per-run timeseries
  - {run_id}/run_{i:02d}_events.csv       phase timestamps per run
  - summary.csv                           peak/mean stats, mean ± std across runs
  - {run_id}_timeline.png                 averaged VRAM / RAM / CPU / Power plot
"""

import json
import subprocess
import time
import gc
import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from src.models.flux import FluxModel
from src.monitoring import ResourceMonitor, cleanup_gpu
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    # "expandable_segments:True,"      # prevents fragmentation on realloc
    "garbage_collection_threshold:0.8,"  # GC triggers earlier
    # "max_split_size_mb:128"          # limits large-block splits
)

# Config
# OUTPUT_DIR = f"results/offload_benchmark_{time.strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR = "results/offload_benchmark_final"

N_RUNS = 3   # how many times to load → generate → unload per model config
PROMPT = "A Entry of a Sushi Restaurant, The text 'OPEN' appears in red neon letters above the door"
WARM_PROMPT = "A cat holding a sign that says 'Offload strategy'"
GEN_PARAMS = {
    "height": 1024,
    "width":  1024,
    "num_inference_steps": 4,
    "guidance_scale": 1.0,
}

# Average Gamer setup
GAMER_VRAM_GB   = 6.0
GAMER_RAM_GB    = 32.0
GAMER_CPU_CORES = 6
GAMER_LABEL     = (
    f"Avg gamer setup: {GAMER_VRAM_GB:.0f} GB VRAM · "
    f"{GAMER_RAM_GB:.0f} GB RAM · {GAMER_CPU_CORES} CPU cores"
)
AVG_VRAM_USER = GAMER_VRAM_GB   # hard VRAM cap for smart offload strategy

_MODEL_ID  = "black-forest-labs/FLUX.2-klein-4B"
_GGUF_REPO = "unsloth/FLUX.2-klein-4B-GGUF"
_QUANTS = [
    ("FLUX2-Klein-FP16",   None),
    ("FLUX2-Klein-Q2_K",   f"{_GGUF_REPO}/flux-2-klein-4b-Q2_K.gguf"),
    ("FLUX2-Klein-Q3_K_M", f"{_GGUF_REPO}/flux-2-klein-4b-Q3_K_M.gguf"),
    # ("FLUX2-Klein-Q4_K_M", f"{_GGUF_REPO}/flux-2-klein-4b-Q4_K_M.gguf"),
    ("FLUX2-Klein-Q5_K_M", f"{_GGUF_REPO}/flux-2-klein-4b-Q5_K_M.gguf"),
    # ("FLUX2-Klein-Q8_0", f"{_GGUF_REPO}/flux-2-klein-4b-Q8_0.gguf"), 
    # ("FLUX2-Klein-Q6_K", f"{_GGUF_REPO}/flux-2-klein-4b-Q6_K.gguf") 
]

# MODELS = [
#     # none          — everything on GPU (fastest baseline, needs full VRAM)
#     # cpu_offload   — diffusers enable_model_cpu_offload()  (component-level)
#     # sequential    — diffusers enable_sequential_cpu_offload()  (sub-module level, min VRAM)
#     #                 ⚠ skipped for GGUF models (incompatible with gguf ops)
#     # group_offload — diffusers enable_group_offload()  (block-level + async streams)
#     # smart         — SmartOffloadManager on transformer, VRAM cap = AVG_VRAM_USER
#     # smart_encode  — SmartOffloadManager on TE + TR sequentially, GPU-accelerated encoding
#     # cpu_only      — entire pipeline on CPU, no GPU involved
#     cfg
#     for name, gguf in _QUANTS
#     for cfg in [
#         {"name": name, "model_id": _MODEL_ID, "gguf_path": gguf, "offload": "none"},
#         {"name": name, "model_id": _MODEL_ID, "gguf_path": gguf, "offload": "cpu_offload"},
#         # sequential only for FP16 — GGUF dequant ops are incompatible with it
#         *([{"name": name, "model_id": _MODEL_ID, "gguf_path": gguf, "offload": "sequential"}]
#           if gguf is None else []),
#         {"name": name, "model_id": _MODEL_ID, "gguf_path": gguf, "offload": "group_offload"},
#         {"name": name, "model_id": _MODEL_ID, "gguf_path": gguf, "offload": "smart",
#          "low_vram_gb": AVG_VRAM_USER},
#         {"name": name, "model_id": _MODEL_ID, "gguf_path": gguf, "offload": "smart_encode",
#          "low_vram_gb": AVG_VRAM_USER},
#         {"name": name, "model_id": _MODEL_ID, "gguf_path": gguf, "offload": "cpu_only",
#          "force_device": "cpu"},
#     ]
# ]
MODELS = [
    {
        "name": name,
        "model_id": _MODEL_ID,
        "gguf_path": gguf,
        "offload": "smart_encode",
        "low_vram_gb": AVG_VRAM_USER,
        "compile": True
    }
    for name, gguf in _QUANTS
]

def runs_for_config(model_cfg: dict) -> int:
    """Use fewer repetitions for very slow strategies."""
    return 1 if model_cfg.get('offload') == 'cpu_only' else N_RUNS
# MODELS = [
#     # {"name": "FLUX2-Klein-FP16", "model_id": _MODEL_ID, "gguf_path": None,
#     #  "offload": "smart_encode", "low_vram_gb": AVG_VRAM_USER},
#     # {"name": "FLUX2-Klein-Q3_K_M", "model_id": _MODEL_ID, "gguf_path": f"{_GGUF_REPO}/flux-2-klein-4b-Q3_K_M.gguf",
#     #  "offload": "cpu_offload"},
#     # {"name": "FLUX2-Klein-Q3_K_M", "model_id": _MODEL_ID, "gguf_path": f"{_GGUF_REPO}/flux-2-klein-4b-Q3_K_M.gguf",
#     #  "offload": "none"},
#     {"name": "FLUX2-Klein-Q3_K_M", "model_id": _MODEL_ID, "gguf_path": f"{_GGUF_REPO}/flux-2-klein-4b-Q3_K_M.gguf",
#      "offload": "smart_encode", "low_vram_gb": AVG_VRAM_USER},
# ]
    

# MODELS=[
#     # {"name": "FLUX2-Klein-FP16", "model_id": _MODEL_ID, "gguf_path": None,
#     #  "offload": "smart_encode", "low_vram_gb": AVG_VRAM_USER},
#     # {"name": "FLUX2-Klein-Q3_K_M", "model_id": _MODEL_ID, "gguf_path": f"{_GGUF_REPO}/flux-2-klein-4b-Q3_K_M.gguf",
#     #  "offload": "none"},
#     {"name": "FLUX2-Klein-Q3_K_M", "model_id": _MODEL_ID, "gguf_path": f"{_GGUF_REPO}/flux-2-klein-4b-Q3_K_M.gguf",
#      "offload": "smart_encode", "low_vram_gb": AVG_VRAM_USER},
# ]
def run_id(model_cfg: dict) -> str:
    """Unique filesystem-safe identifier: name + offload mode."""
    offload = model_cfg['offload']
    name    = model_cfg['name']
    if offload == 'none':
        return name
    suffix = {
        'cpu_offload':   'CPUOff',
        'sequential':    'SeqOff',
        'group_offload': 'GroupOff',
        'smart':         f"Smart{model_cfg.get('low_vram_gb', AVG_VRAM_USER):.0f}GB",
        'smart_encode':  f"SmartEnc{model_cfg.get('low_vram_gb', AVG_VRAM_USER):.0f}GB",
        'cpu_only':      'CPUOnly',
    }[offload]
    if offload == 'smart_encode' and model_cfg.get('compile', False):
        suffix = f"{suffix}_compile"
    return f"{name}__{suffix}"


def interpolate_series(time_arr: np.ndarray, values: np.ndarray,
                       common_t: np.ndarray) -> np.ndarray:
    """Linearly interpolate a time-series onto a common time grid.
    Values outside the original range are filled with the boundary value."""
    return np.interp(common_t, time_arr, values,
                     left=values[0], right=values[-1])

def _vae_flush_cb(pipe, step, timestep, kwargs):
    """Flush the caching-allocator's streaming fragments before VAE decode.

    Called by callback_on_step_end on the last denoising step so the
    ~4 GB of reserved-but-free blocks from the ring-buffer streaming don't
    stack on top of the VAE's activation allocations.
    """
    if step == GEN_PARAMS['num_inference_steps'] - 1 and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return kwargs

# pipeline factory
def _configure_pipeline(descriptor, transformer, device, model_cfg):
    """
    Load and configure the pipeline for the requested offload mode.

    Returns:
        pipe                 — the configured diffusers pipeline
        teardown_fn          — called before `del pipe` to release manager state
        encode_fn(prompt)    — runs text encoding and returns prompt embeds
        gen_fn(gen, embeds)  — runs image generation from prompt embeds
    """
    offload  = model_cfg['offload']
    low_vram = model_cfg.get('low_vram_gb', GAMER_VRAM_GB)

    def _default_encode_fn(pipe_obj, prompt_text):
        exec_dev = getattr(pipe_obj, '_execution_device', torch.device(device))
        with torch.inference_mode():
            prompt_embeds, _ = pipe_obj.encode_prompt(prompt=prompt_text, device=exec_dev)
        return prompt_embeds

    if offload == 'none':
        pipe = descriptor.load_pipeline(transformer).to(device)
        return pipe, lambda: None, \
            lambda prompt_text: _default_encode_fn(pipe, prompt_text), \
            lambda gen, embeds: pipe(
                prompt=None,
                prompt_embeds=embeds,
                generator=gen,
                **GEN_PARAMS,
            ).images[0]

    if offload == 'cpu_offload':
        pipe = descriptor.load_pipeline(transformer)
        pipe.enable_model_cpu_offload()
        return pipe, lambda: None, \
            lambda prompt_text: _default_encode_fn(pipe, prompt_text), \
            lambda gen, embeds: pipe(
                prompt=None,
                prompt_embeds=embeds,
                generator=gen,
                **GEN_PARAMS,
            ).images[0]

    if offload == 'sequential':
        pipe = descriptor.load_pipeline(transformer)
        pipe.enable_sequential_cpu_offload()
        return pipe, lambda: None, \
            lambda prompt_text: _default_encode_fn(pipe, prompt_text), \
            lambda gen, embeds: pipe(
                prompt=None,
                prompt_embeds=embeds,
                generator=gen,
                **GEN_PARAMS,
            ).images[0]

    if offload == 'group_offload':
        from src.offload.pipeline_utils import fix_cpu_text_encoder
        pipe = descriptor.load_pipeline(transformer)
        pipe.transformer.enable_group_offload(
            onload_device=torch.device(device),
            offload_device=torch.device('cpu'),
            offload_type='block_level',
            use_stream=True,
            num_blocks_per_group=1,
        )
        pipe.vae.to(device)
        fix_cpu_text_encoder(pipe, device)
        return pipe, lambda: None, \
            lambda prompt_text: _default_encode_fn(pipe, prompt_text), \
            lambda gen, embeds: pipe(
                prompt=None,
                prompt_embeds=embeds,
                generator=gen,
                **GEN_PARAMS,
            ).images[0]

    if offload == 'smart':
        from src.offload.offload import SmartOffloadManager
        from src.offload.pipeline_utils import fix_cpu_text_encoder
        pipe = descriptor.load_pipeline(transformer)
        pipe.vae.to(device)
        fix_cpu_text_encoder(pipe, device)
        extra_gb = descriptor.activation_headroom_gb(GEN_PARAMS['height'], GEN_PARAMS['width'])
        mgr = SmartOffloadManager(
            transformer, max_vram_gb=low_vram, device=device,
            num_streams=2, extra_reserved_gb=extra_gb,
        )
        mgr.load(force_full_load=False)
        return pipe, mgr.unload, \
            lambda prompt_text: _default_encode_fn(pipe, prompt_text), \
            lambda gen, embeds: pipe(
                prompt=None,
                prompt_embeds=embeds,
                generator=gen,
                **GEN_PARAMS,
                callback_on_step_end=_vae_flush_cb,
                callback_on_step_end_tensor_inputs=[],
            ).images[0]

    if offload == 'smart_encode':
        from src.offload.offload import SmartOffloadManager
        from src.offload.pipeline_utils import fix_execution_device
        pipe = descriptor.load_pipeline(transformer)
        pipe.vae.to(device)
        pipe.text_encoder.to('cpu')
        pipe.transformer.to('cpu')
        if model_cfg.get('compile', False):
            if hasattr(torch, 'compile'):
                # pipe.vae.decode = torch.compile(
                #     pipe.vae.decode,
                #     mode='reduce-overhead',
                #     # fullgraph=True,
                # )
                pipe.vae.decoder.forward = torch.compile(
                    pipe.vae.decoder.forward, 
                    # mode='max-autotune'
                )
                print("  smart_encode: torch.compile enabled for vae.decode")
            else:
                print("  smart_encode: torch.compile requested but unavailable")
        print(
            f"  smart_encode devices: "
            f"vae={next(pipe.vae.parameters()).device}, "
            f"text_encoder={next(pipe.text_encoder.parameters()).device}, "
            f"transformer={next(pipe.transformer.parameters()).device}"
        )


        fix_execution_device(pipe, device)
        exec_dev = pipe._execution_device
        # Phase 1: GPU-accelerated encoding via TE SmartOffloadManager
        te_mgr = SmartOffloadManager(
            pipe.text_encoder, max_vram_gb=low_vram,
            device=device, num_streams=2, extra_reserved_gb=0.4,
        )
        headroom = descriptor.activation_headroom_gb(GEN_PARAMS['height'], GEN_PARAMS['width'])
        print(f"  smart_encode: TE headroom = {headroom:.2f} GB at {GEN_PARAMS['height']}×{GEN_PARAMS['width']}")
        tr_mgr = SmartOffloadManager(
            pipe.transformer, max_vram_gb=low_vram,
            device=device, num_streams=2, extra_reserved_gb=headroom,
        )

        def _smart_encode_teardown():
            tr_mgr.unload()
            te_mgr.unload()
            pipe.text_encoder = None

        def _smart_encode_encode(prompt_text):
            te_mgr.load()
            with torch.inference_mode():
                prompt_embeds, _ = pipe.encode_prompt(prompt=prompt_text, device=exec_dev)
            te_mgr.unload()
            cleanup_gpu()
            return prompt_embeds

        def _smart_encode_gen(gen, embeds):
            tr_mgr.load()
            try:
                img = pipe(
                    prompt=None,
                    prompt_embeds=embeds,
                    generator=gen, **GEN_PARAMS,
                    callback_on_step_end=_vae_flush_cb,
                    callback_on_step_end_tensor_inputs=[],
                ).images[0]
            finally:
                # CRITICAL: Unload TR to make room for the next TE load (warm encode)
                tr_mgr.unload() 
                cleanup_gpu()
            return img
            
        return pipe, _smart_encode_teardown, _smart_encode_encode, _smart_encode_gen

    if offload == 'cpu_only':
        pipe = descriptor.load_pipeline(transformer).to('cpu')
        return pipe, lambda: None, \
            lambda prompt_text: _default_encode_fn(pipe, prompt_text), \
            lambda gen, embeds: pipe(
                prompt=None,
                prompt_embeds=embeds,
                generator=gen,
                **GEN_PARAMS,
            ).images[0]

    raise ValueError(f"Unknown offload mode: {offload!r}")


# Single run: load → monitor → generate → unload → stop monitor

def single_run(model_cfg: dict, run_index: int, seed: int,
               run_dir: Path) -> dict:
    """
    Execute one full lifecycle: monitor starts → model loads
    → encode cold → generate cold → encode warm → generate warm
    → model unloaded → monitor stops.

    Returns a dict with scalar metrics for this run.
    """
    monitor = ResourceMonitor(sample_rate_hz=5.0)   # 5 samples/s
    monitor.start()
    events = {}

    def stamp(name):
        events[name] = time.time() - monitor._start_time

    def _segment_stats(df: pd.DataFrame, start_t: float, end_t: float, time_col: str) -> dict:
        seg = df[(df[time_col] >= start_t) & (df[time_col] <= end_t)]
        if seg.empty:
            return {
                'vram_reserved_peak_mb': np.nan,
                'vram_reserved_mean_mb': np.nan,
                'vram_allocated_peak_mb': np.nan,
                'vram_allocated_mean_mb': np.nan,
                'ram_peak_mb': np.nan,
                'ram_mean_mb': np.nan,
                'cpu_util_peak': np.nan,
                'cpu_util_mean': np.nan,
                'gpu_util_peak': np.nan,
                'gpu_util_mean': np.nan,
                'power_peak_w': np.nan,
                'power_mean_w': np.nan,
                'pcie_tx_peak_kb_s': np.nan,
                'pcie_tx_mean_kb_s': np.nan,
                'pcie_rx_peak_kb_s': np.nan,
                'pcie_rx_mean_kb_s': np.nan,
            }

        def _max_col(col):
            return float(seg[col].max()) if col in seg.columns else np.nan

        def _mean_col(col):
            return float(seg[col].mean()) if col in seg.columns else np.nan

        return {
            'vram_reserved_peak_mb': _max_col('vram_reserved_mb'),
            'vram_reserved_mean_mb': _mean_col('vram_reserved_mb'),
            'vram_allocated_peak_mb': _max_col('vram_allocated_mb'),
            'vram_allocated_mean_mb': _mean_col('vram_allocated_mb'),
            'ram_peak_mb': _max_col('ram_used_mb'),
            'ram_mean_mb': _mean_col('ram_used_mb'),
            'cpu_util_peak': _max_col('cpu_util'),
            'cpu_util_mean': _mean_col('cpu_util'),
            'gpu_util_peak': _max_col('gpu_util'),
            'gpu_util_mean': _mean_col('gpu_util'),
            'power_peak_w': _max_col('power_watts'),
            'power_mean_w': _mean_col('power_watts'),
            'pcie_tx_peak_kb_s': _max_col('pcie_tx_kb_s'),
            'pcie_tx_mean_kb_s': _mean_col('pcie_tx_kb_s'),
            'pcie_rx_peak_kb_s': _max_col('pcie_rx_kb_s'),
            'pcie_rx_mean_kb_s': _mean_col('pcie_rx_kb_s'),
        }

    stamp('load_start')
    device = model_cfg.get('force_device',
                           'cuda' if torch.cuda.is_available() else 'cpu')
    dtype  = torch.bfloat16

    descriptor  = FluxModel(model_cfg['model_id'], gguf_path=model_cfg['gguf_path'], dtype=dtype)
    transformer = descriptor.load_transformer()
    pipe, teardown_fn, encode_fn, gen_fn = _configure_pipeline(descriptor, transformer, device, model_cfg)

    stamp('load_end')
    load_time = events['load_end'] - events['load_start']

    print(f"device : {device}")

    # Cold encode
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    stamp('encode_cold_start')
    t_encode_cold = time.time()
    prompt_embeds = encode_fn(PROMPT)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    encode_cold_time = time.time() - t_encode_cold
    encode_cold_pytorch_allocated_peak_mb = (
        torch.cuda.max_memory_allocated() / 1024 ** 2
        if torch.cuda.is_available() else 0.0
    )
    stamp('encode_cold_end')

    # Cold image
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    stamp('gen_cold_start')
    t0 = time.time()
    gen = torch.Generator(device=device).manual_seed(seed)
    img = gen_fn(gen, prompt_embeds)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    cold_gen_time = time.time() - t0
    cold_pytorch_allocated_peak_mb = (
        torch.cuda.max_memory_allocated() / 1024 ** 2
        if torch.cuda.is_available() else 0.0
    )
    stamp('gen_cold_end')
    img.save(run_dir / f"run_{run_index:02d}_cold.png")

    # Warm encode (re-encode prompt intentionally)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    stamp('encode_warm_start')
    t_encode_warm = time.time()
    prompt_embeds = encode_fn(WARM_PROMPT)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    encode_warm_time = time.time() - t_encode_warm
    encode_warm_pytorch_allocated_peak_mb = (
        torch.cuda.max_memory_allocated() / 1024 ** 2
        if torch.cuda.is_available() else 0.0
    )
    stamp('encode_warm_end')

    # Warm image (same loaded pipeline/components, new seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    stamp('gen_warm_start')
    t1 = time.time()
    gen = torch.Generator(device=device).manual_seed(seed + 10_000)
    img = gen_fn(gen, prompt_embeds)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    warm_gen_time = time.time() - t1
    warm_pytorch_allocated_peak_mb = (
        torch.cuda.max_memory_allocated() / 1024 ** 2
        if torch.cuda.is_available() else 0.0
    )
    stamp('gen_warm_end')
    img.save(run_dir / f"run_{run_index:02d}_warm.png")

    encode_time = encode_cold_time + encode_warm_time
    gen_time = cold_gen_time + warm_gen_time

    # Working Unloading 
    stamp('cleanup_start')
    teardown_fn()
    if hasattr(pipe, '_internal_dict'):
        for key in list(pipe._internal_dict.keys()):
            try:
                setattr(pipe, key, None)
            except Exception:
                pass
    
    for name, child in list(transformer.named_children()):
        try:
            setattr(transformer, name, None)
        except Exception:
            pass
    del pipe, transformer
    cleanup_gpu()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time.sleep(0.05)
    stamp('cleanup_end')
    time.sleep(0.5)
    monitor.stop()
    # save raw timeseries and events for this run
    df = monitor.get_metrics().to_dataframe().reset_index()
    df.to_csv(run_dir / f"run_{run_index:02d}_timeseries.csv", index=False)

    ev_df = pd.DataFrame(list(events.items()), columns=['event', 'time_s'])
    ev_df.to_csv(run_dir / f"run_{run_index:02d}_events.csv", index=False)

    time_col = 'time' if 'time' in df.columns else 'time_s'
    encode_cold_stats = _segment_stats(df, events['encode_cold_start'], events['encode_cold_end'], time_col)
    encode_warm_stats = _segment_stats(df, events['encode_warm_start'], events['encode_warm_end'], time_col)
    gen_cold_stats = _segment_stats(df, events['gen_cold_start'], events['gen_cold_end'], time_col)
    gen_warm_stats = _segment_stats(df, events['gen_warm_start'], events['gen_warm_end'], time_col)

    # scalar summary for this run
    m = monitor.get_metrics()
    return {
        'run':                            run_index,
        'seed':                           seed,
        'load_time_s':                    load_time,
        'encode_cold_time_s':             encode_cold_time,
        'encode_warm_time_s':             encode_warm_time,
        'encode_time_s':                  encode_time,
        'cold_gen_time_s':                cold_gen_time,
        'warm_gen_time_s':                warm_gen_time,
        'gen_time_s':                     gen_time,
        'total_time_s':                   events['cleanup_end'],
        'vram_reserved_peak_mb':          m.vram_reserved_max_mb,
        'vram_reserved_mean_mb':          m.vram_reserved_mean_mb,
        'vram_allocated_peak_mb':         m.vram_allocated_max_mb,
        'vram_allocated_pytorch_peak_mb': max(cold_pytorch_allocated_peak_mb, warm_pytorch_allocated_peak_mb),
        'encode_cold_vram_allocated_pytorch_peak_mb': encode_cold_pytorch_allocated_peak_mb,
        'encode_warm_vram_allocated_pytorch_peak_mb': encode_warm_pytorch_allocated_peak_mb,
        'cold_vram_allocated_pytorch_peak_mb': cold_pytorch_allocated_peak_mb,
        'warm_vram_allocated_pytorch_peak_mb': warm_pytorch_allocated_peak_mb,
        'encode_cold_vram_reserved_peak_mb': encode_cold_stats['vram_reserved_peak_mb'],
        'encode_cold_vram_reserved_mean_mb': encode_cold_stats['vram_reserved_mean_mb'],
        'encode_cold_vram_allocated_peak_mb': encode_cold_stats['vram_allocated_peak_mb'],
        'encode_cold_vram_allocated_mean_mb': encode_cold_stats['vram_allocated_mean_mb'],
        'encode_cold_ram_peak_mb':         encode_cold_stats['ram_peak_mb'],
        'encode_cold_ram_mean_mb':         encode_cold_stats['ram_mean_mb'],
        'encode_cold_cpu_util_peak':       encode_cold_stats['cpu_util_peak'],
        'encode_cold_cpu_util_mean':       encode_cold_stats['cpu_util_mean'],
        'encode_cold_gpu_util_peak':       encode_cold_stats['gpu_util_peak'],
        'encode_cold_gpu_util_mean':       encode_cold_stats['gpu_util_mean'],
        'encode_cold_power_peak_w':        encode_cold_stats['power_peak_w'],
        'encode_cold_power_mean_w':        encode_cold_stats['power_mean_w'],
        'encode_cold_pcie_tx_peak_kb_s':   encode_cold_stats['pcie_tx_peak_kb_s'],
        'encode_cold_pcie_tx_mean_kb_s':   encode_cold_stats['pcie_tx_mean_kb_s'],
        'encode_cold_pcie_rx_peak_kb_s':   encode_cold_stats['pcie_rx_peak_kb_s'],
        'encode_cold_pcie_rx_mean_kb_s':   encode_cold_stats['pcie_rx_mean_kb_s'],
        'encode_warm_vram_reserved_peak_mb': encode_warm_stats['vram_reserved_peak_mb'],
        'encode_warm_vram_reserved_mean_mb': encode_warm_stats['vram_reserved_mean_mb'],
        'encode_warm_vram_allocated_peak_mb': encode_warm_stats['vram_allocated_peak_mb'],
        'encode_warm_vram_allocated_mean_mb': encode_warm_stats['vram_allocated_mean_mb'],
        'encode_warm_ram_peak_mb':         encode_warm_stats['ram_peak_mb'],
        'encode_warm_ram_mean_mb':         encode_warm_stats['ram_mean_mb'],
        'encode_warm_cpu_util_peak':       encode_warm_stats['cpu_util_peak'],
        'encode_warm_cpu_util_mean':       encode_warm_stats['cpu_util_mean'],
        'encode_warm_gpu_util_peak':       encode_warm_stats['gpu_util_peak'],
        'encode_warm_gpu_util_mean':       encode_warm_stats['gpu_util_mean'],
        'encode_warm_power_peak_w':        encode_warm_stats['power_peak_w'],
        'encode_warm_power_mean_w':        encode_warm_stats['power_mean_w'],
        'encode_warm_pcie_tx_peak_kb_s':   encode_warm_stats['pcie_tx_peak_kb_s'],
        'encode_warm_pcie_tx_mean_kb_s':   encode_warm_stats['pcie_tx_mean_kb_s'],
        'encode_warm_pcie_rx_peak_kb_s':   encode_warm_stats['pcie_rx_peak_kb_s'],
        'encode_warm_pcie_rx_mean_kb_s':   encode_warm_stats['pcie_rx_mean_kb_s'],
        'gen_cold_vram_reserved_peak_mb':  gen_cold_stats['vram_reserved_peak_mb'],
        'gen_cold_vram_reserved_mean_mb':  gen_cold_stats['vram_reserved_mean_mb'],
        'gen_cold_vram_allocated_peak_mb': gen_cold_stats['vram_allocated_peak_mb'],
        'gen_cold_vram_allocated_mean_mb': gen_cold_stats['vram_allocated_mean_mb'],
        'gen_cold_ram_peak_mb':            gen_cold_stats['ram_peak_mb'],
        'gen_cold_ram_mean_mb':            gen_cold_stats['ram_mean_mb'],
        'gen_cold_cpu_util_peak':          gen_cold_stats['cpu_util_peak'],
        'gen_cold_cpu_util_mean':          gen_cold_stats['cpu_util_mean'],
        'gen_cold_gpu_util_peak':          gen_cold_stats['gpu_util_peak'],
        'gen_cold_gpu_util_mean':          gen_cold_stats['gpu_util_mean'],
        'gen_cold_power_peak_w':           gen_cold_stats['power_peak_w'],
        'gen_cold_power_mean_w':           gen_cold_stats['power_mean_w'],
        'gen_cold_pcie_tx_peak_kb_s':      gen_cold_stats['pcie_tx_peak_kb_s'],
        'gen_cold_pcie_tx_mean_kb_s':      gen_cold_stats['pcie_tx_mean_kb_s'],
        'gen_cold_pcie_rx_peak_kb_s':      gen_cold_stats['pcie_rx_peak_kb_s'],
        'gen_cold_pcie_rx_mean_kb_s':      gen_cold_stats['pcie_rx_mean_kb_s'],
        'gen_warm_vram_reserved_peak_mb':  gen_warm_stats['vram_reserved_peak_mb'],
        'gen_warm_vram_reserved_mean_mb':  gen_warm_stats['vram_reserved_mean_mb'],
        'gen_warm_vram_allocated_peak_mb': gen_warm_stats['vram_allocated_peak_mb'],
        'gen_warm_vram_allocated_mean_mb': gen_warm_stats['vram_allocated_mean_mb'],
        'gen_warm_ram_peak_mb':            gen_warm_stats['ram_peak_mb'],
        'gen_warm_ram_mean_mb':            gen_warm_stats['ram_mean_mb'],
        'gen_warm_cpu_util_peak':          gen_warm_stats['cpu_util_peak'],
        'gen_warm_cpu_util_mean':          gen_warm_stats['cpu_util_mean'],
        'gen_warm_gpu_util_peak':          gen_warm_stats['gpu_util_peak'],
        'gen_warm_gpu_util_mean':          gen_warm_stats['gpu_util_mean'],
        'gen_warm_power_peak_w':           gen_warm_stats['power_peak_w'],
        'gen_warm_power_mean_w':           gen_warm_stats['power_mean_w'],
        'gen_warm_pcie_tx_peak_kb_s':      gen_warm_stats['pcie_tx_peak_kb_s'],
        'gen_warm_pcie_tx_mean_kb_s':      gen_warm_stats['pcie_tx_mean_kb_s'],
        'gen_warm_pcie_rx_peak_kb_s':      gen_warm_stats['pcie_rx_peak_kb_s'],
        'gen_warm_pcie_rx_mean_kb_s':      gen_warm_stats['pcie_rx_mean_kb_s'],
        'ram_peak_mb':                    m.ram_max_mb,
        'ram_mean_mb':                    m.ram_mean_mb,
        'gpu_util_mean':                  m.gpu_util_mean,
        'gpu_util_max':                   m.gpu_util_max,
        'cpu_util_mean':                  m.cpu_util_mean,
        'cpu_util_max':                   m.cpu_util_max,
        'power_mean_w':                   m.power_mean_watts or 0,
        'power_max_w':                    m.power_max_watts  or 0,
        'pcie_tx_mean_kb_s':              m.pcie_tx_mean_kb_s or 0,
        'pcie_tx_max_kb_s':               m.pcie_tx_max_kb_s  or 0,
        'pcie_rx_mean_kb_s':              m.pcie_rx_mean_kb_s or 0,
        'pcie_rx_max_kb_s':               m.pcie_rx_max_kb_s  or 0,
    }


# Averaged timeline

def plot_averaged_timeline(run_dir: Path, run_results: list[dict],
                           title: str, save_path: Path):
    """
    For each metric (VRAM, RAM, CPU, Power):
      - Interpolate each run's timeseries onto a common grid
      - Plot mean curve with ±1 std shaded band
      - Add vertical lines for averaged phase events
      - Add horizontal dashed lines for avg-gamer hardware limits
    """
    N_RUNS_actual = len(run_results)
    N_POINTS      = 300

    ts_list = []
    ev_list = []
    for i in range(N_RUNS_actual):
        ts_file = run_dir / f"run_{i:02d}_timeseries.csv"
        ev_file = run_dir / f"run_{i:02d}_events.csv"
        if ts_file.exists():
            ts_list.append(pd.read_csv(ts_file))
        if ev_file.exists():
            ev_df = pd.read_csv(ev_file).set_index('event')['time_s']
            ev_list.append(ev_df.to_dict())

    if not ts_list:
        print(f"  ⚠ No timeseries found in {run_dir}")
        return

    # Use median duration as the common x-axis length so a single slow outlier
    # doesn't stretch the plot. (Each run is a fresh subprocess, so all are cold.)
    time_col = 'time' if 'time' in ts_list[0].columns else 'time_s'
    max_t    = float(np.median([df[time_col].iloc[-1] for df in ts_list]))
    common_t = np.linspace(0, max_t, N_POINTS)

    # Each entry: (col_or_special, ylabel, color, fixed_ylim, gamer_limit_mb)
    #   col='vram' → special dual-series (reserved + allocated) subplot
    #   col='pcie' → special dual-series (TX + RX) subplot
    metrics_cfg = [
        ('vram',        'VRAM (MB)',            None,      None,      GAMER_VRAM_GB * 1024),
        ('ram_used_mb', 'RAM / RSS (MB)',        '#3498db', None,      GAMER_RAM_GB  * 1024),
        ('gpu_util',    'GPU Utilisation (%)',   '#2ecc71', (0, 100),  None),
        ('cpu_util',    'CPU Utilisation (%)',   '#f39c12', (0, 100),  None),
        ('power_watts', 'Power (W)',             '#9b59b6', None,      None),
        ('pcie',        'PCIe (MB/s)',           None,      None,      None),
    ]

    def avg_event(key):
        vals = [ev[key] for ev in ev_list if key in ev]
        return np.mean(vals) if vals else None

    def _interp_col(col):
        matrix = []
        for df in ts_list:
            if col not in df.columns:
                continue
            vals = df[col].ffill().fillna(0).to_numpy().astype(float)
            t    = df[time_col].to_numpy().astype(float)
            matrix.append(interpolate_series(t, vals, common_t))
        return np.stack(matrix) if matrix else None

    def _phase_lines(ax):
        # Events listed with: (key, label, linestyle, label_at_top)
        # Alternate labels top/bottom and rotate vertically to reduce overlap.
        events_cfg = [
            ('load_start',    'load start',   '-'),
            ('load_end',      'loaded',       '--'),
            ('encode_cold_start', 'enc cold start', ':'),
            ('encode_cold_end',   'enc cold end',   ':'),
            ('gen_cold_start','cold start',   ':'),
            ('gen_cold_end',  'cold end',     ':'),
            ('encode_warm_start', 'enc warm start', ':'),
            ('encode_warm_end',   'enc warm end',   ':'),
            ('gen_warm_start','warm start',   ':'),
            ('gen_warm_end',  'warm end',     ':'),
            ('cleanup_start', 'cleanup',      '-.'),
            ('cleanup_end',   'cleanup end',  '-.'),
        ]
        ylim = ax.get_ylim()
        yrange = ylim[1] - ylim[0]
        y_top    = ylim[1] - yrange * 0.25
        y_bottom = ylim[0] + yrange * 0.02
        place_top = True
        for ev_key, label_text, ls in events_cfg:
            t_ev = avg_event(ev_key)
            if t_ev is None:
                continue
            ax.axvline(t_ev, color='grey', linewidth=1.1, linestyle=ls, alpha=0.75)
            ypos = y_top if place_top else y_bottom
            va = 'top' if place_top else 'bottom'
            x_offset = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.005
            ax.text(
                t_ev,
                ypos,
                label_text,
                fontsize=7,
                fontweight='bold',
                color='grey',
                va=va,
                ha='left',
                rotation=90,
                rotation_mode='anchor',
            )
            place_top = not place_top

    fig, axes = plt.subplots(len(metrics_cfg), 1,
                             figsize=(14, 4 * len(metrics_cfg)),
                             sharex=True)
    fig.suptitle(f"{title}\n{GAMER_LABEL}",
                 fontsize=13, fontweight='bold', y=1.01)

    for ax, (col, ylabel, color, fixed_ylim, limit_mb) in zip(axes, metrics_cfg):

        if col == 'vram':
            # Dual series: both reserved and allocated on the same axes
            mat_res   = _interp_col('vram_reserved_mb')
            mat_alloc = _interp_col('vram_allocated_mb')
            if mat_res is None:
                ax.set_visible(False)
                continue
            mean_res = mat_res.mean(axis=0)
            std_res  = mat_res.std(axis=0)
            ax.fill_between(common_t, mean_res - std_res, mean_res + std_res,
                            alpha=0.15, color='#e74c3c')
            ax.plot(common_t, mean_res, linewidth=2.2, color='#e74c3c',
                    label='reserved (mean ± std)')
            if mat_alloc is not None:
                mean_alloc = mat_alloc.mean(axis=0)
                ax.plot(common_t, mean_alloc, linewidth=1.8, color='#e67e22',
                        linestyle='--', label='allocated')
            if limit_mb is not None:
                ax.axhline(limit_mb, color='black', linewidth=1.5, linestyle=':',
                           alpha=0.85, label=f'gamer limit ({limit_mb/1024:.0f} GB)')
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(True, alpha=0.25, linestyle='--')
            ax.set_axisbelow(True)
            ax.legend(fontsize=9, loc='upper right')
            _phase_lines(ax)
            continue

        if col == 'pcie':
            mat_tx = _interp_col('pcie_tx_kb_s')
            mat_rx = _interp_col('pcie_rx_kb_s')
            if mat_tx is None and mat_rx is None:
                ax.set_visible(False)
                continue
            if mat_tx is not None:
                mean_tx = mat_tx.mean(axis=0) / 1024  # KB/s → MB/s
                std_tx  = mat_tx.std(axis=0)  / 1024
                ax.fill_between(common_t, mean_tx - std_tx, mean_tx + std_tx,
                                alpha=0.15, color='#8e44ad')
                ax.plot(common_t, mean_tx, linewidth=2.0, color='#8e44ad',
                        label='TX (mean ± std)')
            if mat_rx is not None:
                mean_rx = mat_rx.mean(axis=0) / 1024
                std_rx  = mat_rx.std(axis=0)  / 1024
                ax.fill_between(common_t, mean_rx - std_rx, mean_rx + std_rx,
                                alpha=0.15, color='#2980b9')
                ax.plot(common_t, mean_rx, linewidth=2.0, color='#2980b9',
                        linestyle='--', label='RX (mean ± std)')
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(True, alpha=0.25, linestyle='--')
            ax.set_axisbelow(True)
            ax.legend(fontsize=9, loc='upper right')
            _phase_lines(ax)
            continue

        # Generic single-series subplot
        mat = _interp_col(col)
        if mat is None:
            ax.set_visible(False)
            continue
        mean = mat.mean(axis=0)
        std  = mat.std(axis=0)

        ax.fill_between(common_t, mean - std, mean + std,
                        alpha=0.20, color=color, linewidth=0)
        ax.plot(common_t, mean, linewidth=2.2, color=color, label='mean ± 1 std')

        if limit_mb is not None:
            ax.axhline(limit_mb, color='black', linewidth=1.5, linestyle=':',
                       alpha=0.85, label=f'gamer limit ({limit_mb/1024:.0f} GB)')

        ax.set_ylabel(ylabel, fontsize=12)
        if fixed_ylim is not None:
            ax.set_ylim(*fixed_ylim)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, loc='upper right')
        _phase_lines(ax)

    axes[-1].set_xlabel('Time (s)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved: {save_path.name}")


# Per-model benchmark

def _collect_and_plot(model_cfg: dict, output_dir: Path) -> list[dict]:
    """Collect scalar JSONs written by per-run workers, plot, return rows."""
    rid     = run_id(model_cfg)
    run_dir = output_dir / rid

    run_results = []
    n_runs = runs_for_config(model_cfg)
    for i in range(n_runs):
        p = run_dir / f"scalar_{i:02d}.json"
        if p.exists():
            run_results.append(json.loads(p.read_text(encoding="utf-8")))
        else:
            print(f"  ⚠ Missing scalar for {rid} run {i}")

    if not run_results:
        return []

    runs_df = pd.DataFrame(run_results)
    runs_df['model']   = model_cfg['name']
    runs_df['offload'] = model_cfg['offload']
    runs_df['run_id']  = rid
    runs_df.to_csv(run_dir / 'runs.csv', index=False)

    title = (f"Resource Usage — {rid}\n"
             f"(mean ± std over {len(run_results)} runs, fresh subprocess each)")
    plot_averaged_timeline(run_dir, run_results, title,
                           save_path=output_dir / f"{rid}_timeline.png")

    return [dict(r, model=model_cfg['name'],
                    offload=model_cfg['offload'],
                    run_id=rid) for r in run_results]


def _merge_with_existing_runs(output_dir: Path,
                              new_results: list[dict],
                              rerun_run_ids: set[str]) -> pd.DataFrame:
    """keep old rows for untouched configs and replace rows for rerun configs."""
    new_df = pd.DataFrame(new_results)
    all_runs_path = output_dir / 'all_runs.csv'

    if all_runs_path.exists():
        existing_df = pd.read_csv(all_runs_path)
        if 'run_id' in existing_df.columns:
            existing_df = existing_df[~existing_df['run_id'].isin(rerun_run_ids)]
            if new_df.empty:
                return existing_df.reset_index(drop=True)
            return pd.concat([existing_df, new_df], ignore_index=True, sort=False)

    return new_df.reset_index(drop=True)


# Main

def _worker_mode(rid: str, run_index: int, output_dir: Path):
    """Run ONE lifecycle in this fresh subprocess and save scalar + timeseries."""
    model_cfg = next(m for m in MODELS if run_id(m) == rid)
    run_dir   = output_dir / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = 42 + run_index
    print(f"  Running {rid}  run {run_index}  (seed={seed})", flush=True)
    r = single_run(model_cfg, run_index, seed, run_dir)
    print(f"  load={r['load_time_s']:.1f}s  enc_cold={r['encode_cold_time_s']:.1f}s  "
          f"enc_warm={r['encode_warm_time_s']:.1f}s  "
          f"cold={r['cold_gen_time_s']:.1f}s  warm={r['warm_gen_time_s']:.1f}s  "
          f"gen_total={r['gen_time_s']:.1f}s  "
          f"VRAM_res={r['vram_reserved_peak_mb']:.0f}MB  "
          f"VRAM_alloc={r['vram_allocated_peak_mb']:.0f}MB", flush=True)
    scalar_path = run_dir / f"scalar_{run_index:02d}.json"
    scalar_path.write_text(json.dumps(r, default=str), encoding="utf-8")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--_worker",  default=None, metavar="RUN_ID",
                        help=argparse.SUPPRESS)
    parser.add_argument("--_run",     default=None, type=int, dest="_run",
                        help=argparse.SUPPRESS)
    parser.add_argument("--_out-dir", default=None, dest="_out_dir",
                        help=argparse.SUPPRESS)
    parser.add_argument("--only-offload", default=None, dest="only_offload",
                        metavar="STRATEGY",
                        help="Run only configs with this offload mode (e.g. cpu_only)")
    args = parser.parse_args()

    if args._worker is not None:
        _worker_mode(args._worker, args._run, Path(args._out_dir))
        return

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("OFFLOAD STRATEGY BENCHMARK")
    print("=" * 70)
    print(f"Configs: {len(MODELS)}")
    print(f"Runs:    {N_RUNS} per config  (one fresh subprocess per run)")
    print(f"Prompt:  {PROMPT[:80]}...")
    print(f"Output:  {output_dir}/")
    print("=" * 70)

    # One subprocess per (config, run_index) → true cold measurement each time
    models_to_run = MODELS
    if args.only_offload:
        models_to_run = [m for m in MODELS if m['offload'] == args.only_offload]
        if not models_to_run:
            print(f"\n[X] No configs found with offload='{args.only_offload}'")
            print(f"   Valid strategies: {sorted({m['offload'] for m in MODELS})}")
            return
        print(f"\nFiltering to offload='{args.only_offload}': {len(models_to_run)} configs")

    all_results = []
    for model_cfg in models_to_run:
        rid = run_id(model_cfg)
        n_runs = runs_for_config(model_cfg)
        print(f"\n{'='*70}")
        print(f"  {rid}  ({n_runs} runs)")
        print(f"{'='*70}")
        for run_index in range(n_runs):
            cmd = [
                sys.executable, __file__,
                "--_worker",  rid,
                "--_run",     str(run_index),
                "--_out-dir", str(output_dir),
            ]
            print(f"\n  Spawning subprocess for {rid}  run {run_index} …", flush=True)
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                print(f"[X] Subprocess exited with code {result.returncode}")

        # Plot immediately after all runs for this config are done
        all_results.extend(_collect_and_plot(model_cfg, output_dir))

    rerun_run_ids = {run_id(m) for m in models_to_run}
    full_df = _merge_with_existing_runs(output_dir, all_results, rerun_run_ids)
    if full_df.empty:
        print("\nNo results collected.")
        return

    full_df.to_csv(output_dir / 'all_runs.csv', index=False)
    print(f"\n✓ all_runs.csv saved ({len(full_df)} rows)")

    numeric_cols = [
        'load_time_s', 'encode_cold_time_s', 'encode_warm_time_s', 'encode_time_s',
        'cold_gen_time_s', 'warm_gen_time_s', 'gen_time_s', 'total_time_s',
        'vram_reserved_peak_mb', 'vram_reserved_mean_mb',
        'vram_allocated_peak_mb', 'vram_allocated_pytorch_peak_mb',
        'encode_cold_vram_allocated_pytorch_peak_mb', 'encode_warm_vram_allocated_pytorch_peak_mb',
        'cold_vram_allocated_pytorch_peak_mb', 'warm_vram_allocated_pytorch_peak_mb',
        'encode_cold_vram_reserved_peak_mb', 'encode_cold_vram_reserved_mean_mb',
        'encode_cold_vram_allocated_peak_mb', 'encode_cold_vram_allocated_mean_mb',
        'encode_cold_ram_peak_mb', 'encode_cold_ram_mean_mb',
        'encode_cold_cpu_util_peak', 'encode_cold_cpu_util_mean',
        'encode_cold_gpu_util_peak', 'encode_cold_gpu_util_mean',
        'encode_cold_power_peak_w', 'encode_cold_power_mean_w',
        'encode_cold_pcie_tx_peak_kb_s', 'encode_cold_pcie_tx_mean_kb_s',
        'encode_cold_pcie_rx_peak_kb_s', 'encode_cold_pcie_rx_mean_kb_s',
        'encode_warm_vram_reserved_peak_mb', 'encode_warm_vram_reserved_mean_mb',
        'encode_warm_vram_allocated_peak_mb', 'encode_warm_vram_allocated_mean_mb',
        'encode_warm_ram_peak_mb', 'encode_warm_ram_mean_mb',
        'encode_warm_cpu_util_peak', 'encode_warm_cpu_util_mean',
        'encode_warm_gpu_util_peak', 'encode_warm_gpu_util_mean',
        'encode_warm_power_peak_w', 'encode_warm_power_mean_w',
        'encode_warm_pcie_tx_peak_kb_s', 'encode_warm_pcie_tx_mean_kb_s',
        'encode_warm_pcie_rx_peak_kb_s', 'encode_warm_pcie_rx_mean_kb_s',
        'gen_cold_vram_reserved_peak_mb', 'gen_cold_vram_reserved_mean_mb',
        'gen_cold_vram_allocated_peak_mb', 'gen_cold_vram_allocated_mean_mb',
        'gen_cold_ram_peak_mb', 'gen_cold_ram_mean_mb',
        'gen_cold_cpu_util_peak', 'gen_cold_cpu_util_mean',
        'gen_cold_gpu_util_peak', 'gen_cold_gpu_util_mean',
        'gen_cold_power_peak_w', 'gen_cold_power_mean_w',
        'gen_cold_pcie_tx_peak_kb_s', 'gen_cold_pcie_tx_mean_kb_s',
        'gen_cold_pcie_rx_peak_kb_s', 'gen_cold_pcie_rx_mean_kb_s',
        'gen_warm_vram_reserved_peak_mb', 'gen_warm_vram_reserved_mean_mb',
        'gen_warm_vram_allocated_peak_mb', 'gen_warm_vram_allocated_mean_mb',
        'gen_warm_ram_peak_mb', 'gen_warm_ram_mean_mb',
        'gen_warm_cpu_util_peak', 'gen_warm_cpu_util_mean',
        'gen_warm_gpu_util_peak', 'gen_warm_gpu_util_mean',
        'gen_warm_power_peak_w', 'gen_warm_power_mean_w',
        'gen_warm_pcie_tx_peak_kb_s', 'gen_warm_pcie_tx_mean_kb_s',
        'gen_warm_pcie_rx_peak_kb_s', 'gen_warm_pcie_rx_mean_kb_s',
        'ram_peak_mb', 'ram_mean_mb',
        'gpu_util_mean', 'cpu_util_mean',
        'power_mean_w', 'power_max_w',
        'pcie_tx_mean_kb_s', 'pcie_tx_max_kb_s',
        'pcie_rx_mean_kb_s', 'pcie_rx_max_kb_s',
    ]

    summary_rows = []
    for rid_val, grp in full_df.groupby('run_id'):
        row = {'run_id': rid_val,
               'model':  grp['model'].iloc[0],
               'offload': grp['offload'].iloc[0],
               'n_runs':  len(grp)}
        for col in numeric_cols:
            if col in grp.columns:
                row[f'{col}_mean'] = grp[col].mean()
                row[f'{col}_std']  = grp[col].std()
                if 'vram_allocated' in col:
                    std = row[f'{col}_std']
                    n = len(grp)
                    row[f'{col}_ci95'] = (
                        1.96 * std / math.sqrt(n)
                        if pd.notna(std) and n > 0 else np.nan
                    )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / 'summary.csv', index=False)
    print(f"✓ summary.csv saved ({len(summary_df)} model configs)")

    print("\n" + "=" * 70)
    print("SUMMARY  (mean ± std across runs)")
    print("=" * 70)
    cols_to_show = ['run_id', 'load_time_s_mean', 'load_time_s_std',
                    'encode_cold_time_s_mean', 'encode_cold_time_s_std',
                    'encode_warm_time_s_mean', 'encode_warm_time_s_std',
                    'encode_time_s_mean', 'encode_time_s_std',
                    'cold_gen_time_s_mean', 'cold_gen_time_s_std',
                    'warm_gen_time_s_mean', 'warm_gen_time_s_std',
                    'gen_time_s_mean', 'gen_time_s_std',
                    'vram_reserved_peak_mb_mean', 'vram_reserved_peak_mb_std',
                    'vram_allocated_peak_mb_mean']
    cols_to_show = [c for c in cols_to_show if c in summary_df.columns]
    print(summary_df[cols_to_show].to_string(index=False))

    print(f"\n✓ Done.  Results in: {output_dir}/")


if __name__ == '__main__':
    main()
