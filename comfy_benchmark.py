"""Benchmark FLUX.2-Klein FP16 with ComfyUI low-VRAM settings."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

COMFYUI_ROOT    = Path("/home/ensta/ensta-gassem/ComfyUI")
COMFYUI_PYTHON  = sys.executable   # interpreter that has ComfyUI deps

UNET_NAME = "flux-2-klein-4b.safetensors"   # models/diffusion_models/
CLIP_NAME = "qwen_3_4b.safetensors"          # models/text_encoders/
VAE_NAME  = "flux2-vae.safetensors"          # models/vae/

# VRAM cap: L40S has 48 GB; reserve 42 → ComfyUI sees ≤ 6 GB usable
GAMER_VRAM_GB   = 6.0
GAMER_RAM_GB    = 32.0
GAMER_CPU_CORES = 6
RESERVE_VRAM_GB = 42.0

GAMER_LABEL = (
    f"Avg gamer setup: {GAMER_VRAM_GB:.0f} GB VRAM · "
    f"{GAMER_RAM_GB:.0f} GB RAM · {GAMER_CPU_CORES} CPU cores"
)

OUTPUT_DIR = "results/offload_benchmark_final"
N_RUNS     = 3

PROMPT      = ("A Entry of a Sushi Restaurant, "
               "The text 'OPEN' appears in red neon letters above the door")
WARM_PROMPT = "A cat holding a sign that says 'Offload strategy'"

GEN_PARAMS = {
    "height":   1024,
    "width":    1024,
    "steps":    4,
    "cfg":      1.0,
    "sampler":  "euler",
    "scheduler":"simple",
}

_RUN_ID = f"FLUX2-Klein-FP16__ComfyUI_lowvram{GAMER_VRAM_GB:.0f}GB"

MODEL_CFG = {
    "name":        "FLUX2-Klein-FP16",
    "offload":     "comfyui_lowvram",
    "run_id":      _RUN_ID,
    "low_vram_gb": GAMER_VRAM_GB,
}


def _setup_comfyui_vram_and_paths():
    """
    Setup ComfyUI VRAM arguments and paths in the CORRECT order:
    1. Replace sys.argv with ONLY ComfyUI-recognized arguments
    2. Add ComfyUI to sys.path
    3. Import comfy.options and call enable_args_parsing()
    4. Now safe to import other ComfyUI modules
    """
    comfyui_argv = [sys.argv[0]]
    
    if "--lowvram" not in comfyui_argv:
        comfyui_argv.append("--lowvram")
    if "--disable-smart-memory" not in comfyui_argv:
        comfyui_argv.append("--disable-smart-memory")
    if "--reserve-vram" not in " ".join(comfyui_argv):
        comfyui_argv.extend(["--reserve-vram", str(RESERVE_VRAM_GB)])
    
    original_argv = sys.argv.copy()
    sys.argv = comfyui_argv
    
    root = str(COMFYUI_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    
    import comfy.options
    comfy.options.enable_args_parsing()
    
    comfy_sub = str((COMFYUI_ROOT / "comfy").resolve())
    if comfy_sub not in sys.path:
        sys.path.insert(0, comfy_sub)
    
    import logging
    logging.info("ComfyUI VRAM setup (FIXED):")
    logging.info(f"  Original sys.argv: {original_argv}")
    logging.info(f"  ComfyUI sys.argv: {sys.argv}")
    logging.info(f"  args_parsing enabled: True")
    logging.info(f"  --lowvram: True")
    logging.info(f"  --reserve-vram: {RESERVE_VRAM_GB} GB")
    logging.info(f"  --disable-smart-memory: True")


# src/ monitoring import
sys.path.insert(0, str(Path(__file__).parent))

import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.monitoring import ResourceMonitor, cleanup_gpu  # type: ignore


def _generate_with_timing(prompt: str, seed: int, output_path: Path, 
                          events: dict, t_start: float) -> dict:
    """
    Run one full encode → sample → VAE-decode cycle with REAL timing.
    
    Returns dict with:
        - timing for each phase (text_encoder_load, encode, diffusion_load, sample, vae_load, vae_decode)
        - pytorch memory peak for this generation
    """
    import folder_paths
    import comfy.model_management as mm
    import comfy.sd as sd
    import comfy.sample as sample_mod
    import comfy.utils as utils
    from comfy.cli_args import args as cli_args
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    def stamp(name: str):
        events[name] = time.time() - t_start

    timing = {}
    
    logger.info(f"ComfyUI VRAM settings:")
    logger.info(f"  vram_state: {mm.vram_state}")
    logger.info(f"  CLI lowvram: {cli_args.lowvram}")
    logger.info(f"  CLI reserve_vram: {cli_args.reserve_vram}")
    from comfy.model_management import EXTRA_RESERVED_VRAM
    logger.info(f"  EXTRA_RESERVED_VRAM: {EXTRA_RESERVED_VRAM / (1024**3):.2f} GB")

    unet_name = UNET_NAME
    is_distilled = "flux-2-klein-4b" in unet_name.lower() and "base" not in unet_name.lower()

    unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
    clip_path = folder_paths.get_full_path_or_raise("text_encoders",    CLIP_NAME)
    vae_path  = folder_paths.get_full_path_or_raise("vae",              VAE_NAME)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        # 1. TEXT ENCODER LOAD
        stamp("text_encoder_load_start")
        t0 = time.time()
        logger.info("Loading text encoder (Qwen3-4B)...")
        clip = sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=sd.CLIPType.FLUX2,
        )
        timing['text_encoder_load_time_s'] = time.time() - t0
        stamp("text_encoder_load_end")
        logger.info(f"Text encoder loaded in {timing['text_encoder_load_time_s']:.2f}s")

        # 2. PROMPT ENCODING
        stamp("encode_start")
        t0 = time.time()
        logger.info(f"Encoding prompt: {prompt[:60]}...")
        tokens   = clip.tokenize(prompt)
        positive = clip.encode_from_tokens_scheduled(tokens)

        if is_distilled:
            negative = []
            for t in positive:
                d = t[1].copy()
                if d.get("pooled_output") is not None:
                    d["pooled_output"] = torch.zeros_like(d["pooled_output"])
                if d.get("conditioning_lyrics") is not None:
                    d["conditioning_lyrics"] = torch.zeros_like(d["conditioning_lyrics"])
                negative.append([torch.zeros_like(t[0]), d])
        else:
            empty_tokens = clip.tokenize("")
            negative     = clip.encode_from_tokens_scheduled(empty_tokens)
        
        timing['encode_time_s'] = time.time() - t0
        stamp("encode_end")
        logger.info(f"Prompt encoded in {timing['encode_time_s']:.2f}s")

        # 3. DIFFUSION MODEL LOAD
        stamp("diffusion_load_start")
        t0 = time.time()
        logger.info("Loading diffusion model...")
        model_patcher = sd.load_diffusion_model(unet_path, model_options={})
        timing['diffusion_load_time_s'] = time.time() - t0
        stamp("diffusion_load_end")
        logger.info(f"Diffusion model loaded in {timing['diffusion_load_time_s']:.2f}s")

        # 4. CREATE LATENT
        latent = torch.zeros(
            [1, 128, GEN_PARAMS["height"] // 16, GEN_PARAMS["width"] // 16],
            device=mm.intermediate_device(),
        )
        latent = sample_mod.fix_empty_latent_channels(model_patcher, latent)

        # 5. SAMPLING
        stamp("sample_start")
        t0 = time.time()
        logger.info(f"Sampling {GEN_PARAMS['steps']} steps, "
                    f"CFG={GEN_PARAMS['cfg']}, "
                    f"sampler={GEN_PARAMS['sampler']}, "
                    f"scheduler={GEN_PARAMS['scheduler']}...")
        noise   = sample_mod.prepare_noise(latent, seed, None)
        samples = sample_mod.sample(
            model_patcher,
            noise,
            GEN_PARAMS["steps"],
            GEN_PARAMS["cfg"],
            GEN_PARAMS["sampler"],
            GEN_PARAMS["scheduler"],
            positive,
            negative,
            latent,
            denoise=1.0,
            disable_pbar=False,
            seed=seed,
        )
        timing['sample_time_s'] = time.time() - t0
        stamp("sample_end")
        logger.info(f"Sampling complete in {timing['sample_time_s']:.2f}s")

        # 6. VAE LOAD
        stamp("vae_load_start")
        t0 = time.time()
        logger.info("Loading VAE...")
        vae_sd, vae_meta = utils.load_torch_file(vae_path, return_metadata=True)
        vae = sd.VAE(sd=vae_sd, metadata=vae_meta)
        timing['vae_load_time_s'] = time.time() - t0
        stamp("vae_load_end")
        logger.info(f"VAE loaded in {timing['vae_load_time_s']:.2f}s")

        # 7. VAE DECODE
        stamp("vae_decode_start")
        t0 = time.time()
        logger.info("Decoding latent...")
        images = vae.decode(samples)
        timing['vae_decode_time_s'] = time.time() - t0
        stamp("vae_decode_end")
        logger.info(f"VAE decode complete in {timing['vae_decode_time_s']:.2f}s")

        # 8. SAVE IMAGE
        stamp("save_start")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img_np = images[0].cpu().numpy()
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        if img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
        pil_img = Image.fromarray(img_np)
        pnginfo = PngInfo()
        pnginfo.add_text("prompt",    prompt)
        pnginfo.add_text("seed",      str(seed))
        pnginfo.add_text("steps",     str(GEN_PARAMS["steps"]))
        pnginfo.add_text("cfg",       str(GEN_PARAMS["cfg"]))
        pnginfo.add_text("sampler",   GEN_PARAMS["sampler"])
        pnginfo.add_text("scheduler", GEN_PARAMS["scheduler"])
        pil_img.save(str(output_path), pnginfo=pnginfo)
        stamp("save_end")
        logger.info(f"Image saved: {output_path}")

    timing['pytorch_alloc_peak_mb'] = (
        torch.cuda.max_memory_allocated() / 1024 ** 2
        if torch.cuda.is_available() else 0.0
    )

    timing['total_time_s'] = (
        timing['text_encoder_load_time_s'] +
        timing['encode_time_s'] +
        timing['diffusion_load_time_s'] +
        timing['sample_time_s'] +
        timing['vae_load_time_s'] +
        timing['vae_decode_time_s']
    )

    return timing


def _segment_stats(df: pd.DataFrame, t0: float, t1: float,
                   time_col: str) -> dict:
    seg = df[(df[time_col] >= t0) & (df[time_col] <= t1)]
    nan = float("nan")
    if seg.empty:
        return {k: nan for k in [
            "vram_reserved_peak_mb", "vram_reserved_mean_mb",
            "vram_allocated_peak_mb", "vram_allocated_mean_mb",
            "ram_peak_mb", "ram_mean_mb",
            "cpu_util_peak", "cpu_util_mean",
            "gpu_util_peak", "gpu_util_mean",
            "power_peak_w", "power_mean_w",
            "pcie_tx_peak_kb_s", "pcie_tx_mean_kb_s",
            "pcie_rx_peak_kb_s", "pcie_rx_mean_kb_s",
        ]}

    def mx(c): return float(seg[c].max())  if c in seg.columns else nan
    def mn(c): return float(seg[c].mean()) if c in seg.columns else nan

    return {
        "vram_reserved_peak_mb":  mx("vram_reserved_mb"),
        "vram_reserved_mean_mb":  mn("vram_reserved_mb"),
        "vram_allocated_peak_mb": mx("vram_allocated_mb"),
        "vram_allocated_mean_mb": mn("vram_allocated_mb"),
        "ram_peak_mb":            mx("ram_used_mb"),
        "ram_mean_mb":            mn("ram_used_mb"),
        "cpu_util_peak":          mx("cpu_util"),
        "cpu_util_mean":          mn("cpu_util"),
        "gpu_util_peak":          mx("gpu_util"),
        "gpu_util_mean":          mn("gpu_util"),
        "power_peak_w":           mx("power_watts"),
        "power_mean_w":           mn("power_watts"),
        "pcie_tx_peak_kb_s":      mx("pcie_tx_kb_s"),
        "pcie_tx_mean_kb_s":      mn("pcie_tx_mean_kb_s"),
        "pcie_rx_peak_kb_s":      mx("pcie_rx_kb_s"),
        "pcie_rx_mean_kb_s":      mn("pcie_rx_kb_s"),
    }


def single_run(run_index: int, seed: int, run_dir: Path) -> dict:
    """
    Full lifecycle with accurate timing for each phase.
    """
    monitor = ResourceMonitor(sample_rate_hz=5.0)
    monitor.start()
    t_start = monitor._start_time
    events: dict[str, float] = {}

    def stamp(name: str):
        events[name] = time.time() - t_start

    timing_results = {}

    stamp("cold_start")
    cold_timing = _generate_with_timing(
        PROMPT, seed, 
        run_dir / f"run_{run_index:02d}_cold.png",
        events, t_start
    )
    stamp("cold_end")
    
    # Store cold timing with prefix
    for k, v in cold_timing.items():
        timing_results[f"cold_{k}"] = v

    stamp("warm_start")
    warm_timing = _generate_with_timing(
        WARM_PROMPT, seed + 10_000, 
        run_dir / f"run_{run_index:02d}_warm.png",
        events, t_start
    )
    stamp("warm_end")
    
    # Store warm timing with prefix
    for k, v in warm_timing.items():
        timing_results[f"warm_{k}"] = v

    stamp("cleanup_start")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(0.5)
    stamp("cleanup_end")

    monitor.stop()

    df = monitor.get_metrics().to_dataframe().reset_index()
    df.to_csv(run_dir / f"run_{run_index:02d}_timeseries.csv", index=False)
    ev_df = pd.DataFrame(list(events.items()), columns=["event", "time_s"])
    ev_df.to_csv(run_dir / f"run_{run_index:02d}_events.csv", index=False)

    time_col = "time" if "time" in df.columns else "time_s"

    te_load_cold = _segment_stats(df, events.get("text_encoder_load_start", 0), 
                                   events.get("text_encoder_load_end", 0), time_col)
    encode_cold = _segment_stats(df, events.get("encode_start", 0), 
                                  events.get("encode_end", 0), time_col)
    diff_load_cold = _segment_stats(df, events.get("diffusion_load_start", 0), 
                                     events.get("diffusion_load_end", 0), time_col)
    sample_cold = _segment_stats(df, events.get("sample_start", 0), 
                                  events.get("sample_end", 0), time_col)
    vae_load_cold = _segment_stats(df, events.get("vae_load_start", 0), 
                                    events.get("vae_load_end", 0), time_col)
    vae_decode_cold = _segment_stats(df, events.get("vae_decode_start", 0), 
                                      events.get("vae_decode_end", 0), time_col)
    
    te_load_warm = _segment_stats(df, events.get("warm_start", 0), 
                                   events.get("text_encoder_load_end", 0), time_col)
    encode_warm_seg = _segment_stats(df, events.get("warm_start", 0), 
                                      events.get("encode_end", 0), time_col)
    diff_load_warm = _segment_stats(df, events.get("diffusion_load_start", 0), 
                                     events.get("diffusion_load_end", 0), time_col)
    sample_warm = _segment_stats(df, events.get("sample_start", 0), 
                                  events.get("sample_end", 0), time_col)

    m = monitor.get_metrics()

    # Build result dict with REAL timing
    result = {
        "run":  run_index,
        "seed": seed,
        
        # REAL TIMING (seconds) - from _generate_with_timing
        # Cold run
        "cold_text_encoder_load_time_s": cold_timing['text_encoder_load_time_s'],
        "cold_encode_time_s":            cold_timing['encode_time_s'],
        "cold_diffusion_load_time_s":    cold_timing['diffusion_load_time_s'],
        "cold_sample_time_s":            cold_timing['sample_time_s'],
        "cold_vae_load_time_s":          cold_timing['vae_load_time_s'],
        "cold_vae_decode_time_s":        cold_timing['vae_decode_time_s'],
        "cold_total_time_s":             cold_timing['total_time_s'],
        
        # Warm run  
        "warm_text_encoder_load_time_s": warm_timing['text_encoder_load_time_s'],
        "warm_encode_time_s":            warm_timing['encode_time_s'],
        "warm_diffusion_load_time_s":    warm_timing['diffusion_load_time_s'],
        "warm_sample_time_s":            warm_timing['sample_time_s'],
        "warm_vae_load_time_s":          warm_timing['vae_load_time_s'],
        "warm_vae_decode_time_s":        warm_timing['vae_decode_time_s'],
        "warm_total_time_s":             warm_timing['total_time_s'],
        
        # Aggregated timing
        "text_encoder_load_time_s": cold_timing['text_encoder_load_time_s'] + warm_timing['text_encoder_load_time_s'],
        "encode_time_s":            cold_timing['encode_time_s'] + warm_timing['encode_time_s'],
        "diffusion_load_time_s":    cold_timing['diffusion_load_time_s'] + warm_timing['diffusion_load_time_s'],
        "sample_time_s":            cold_timing['sample_time_s'] + warm_timing['sample_time_s'],
        "vae_load_time_s":          cold_timing['vae_load_time_s'] + warm_timing['vae_load_time_s'],
        "vae_decode_time_s":        cold_timing['vae_decode_time_s'] + warm_timing['vae_decode_time_s'],
        "gen_time_s":               cold_timing['total_time_s'] + warm_timing['total_time_s'],
        "total_time_s":             events["cleanup_end"],
        
        # Legacy field names (for compatibility with existing analysis)
        "load_time_s":        cold_timing['text_encoder_load_time_s'] + cold_timing['diffusion_load_time_s'],
        "encode_cold_time_s": cold_timing['encode_time_s'],
        "encode_warm_time_s": warm_timing['encode_time_s'],
        "cold_gen_time_s":    cold_timing['sample_time_s'] + cold_timing['vae_decode_time_s'],
        "warm_gen_time_s":    warm_timing['sample_time_s'] + warm_timing['vae_decode_time_s'],
        
        # VRAM — measured in this process
        "vram_reserved_peak_mb":          m.vram_reserved_max_mb,
        "vram_reserved_mean_mb":          m.vram_reserved_mean_mb,
        "vram_allocated_peak_mb":         m.vram_allocated_max_mb,
        "vram_allocated_pytorch_peak_mb": max(cold_timing['pytorch_alloc_peak_mb'],
                                              warm_timing['pytorch_alloc_peak_mb']),
        "cold_vram_allocated_pytorch_peak_mb":  cold_timing['pytorch_alloc_peak_mb'],
        "warm_vram_allocated_pytorch_peak_mb":  warm_timing['pytorch_alloc_peak_mb'],
        
        # Segment stats - encode phases
        "encode_cold_vram_reserved_peak_mb":  encode_cold["vram_reserved_peak_mb"],
        "encode_cold_vram_reserved_mean_mb":  encode_cold["vram_reserved_mean_mb"],
        "encode_cold_vram_allocated_peak_mb": encode_cold["vram_allocated_peak_mb"],
        "encode_cold_vram_allocated_mean_mb": encode_cold["vram_allocated_mean_mb"],
        "encode_cold_ram_peak_mb":            encode_cold["ram_peak_mb"],
        "encode_cold_ram_mean_mb":            encode_cold["ram_mean_mb"],
        "encode_cold_cpu_util_peak":          encode_cold["cpu_util_peak"],
        "encode_cold_cpu_util_mean":          encode_cold["cpu_util_mean"],
        "encode_cold_gpu_util_peak":          encode_cold["gpu_util_peak"],
        "encode_cold_gpu_util_mean":          encode_cold["gpu_util_mean"],
        
        "encode_warm_vram_reserved_peak_mb":  encode_warm_seg["vram_reserved_peak_mb"],
        "encode_warm_vram_reserved_mean_mb":  encode_warm_seg["vram_reserved_mean_mb"],
        "encode_warm_vram_allocated_peak_mb": encode_warm_seg["vram_allocated_peak_mb"],
        "encode_warm_vram_allocated_mean_mb": encode_warm_seg["vram_allocated_mean_mb"],
        
        # Segment stats - sample phases
        "sample_cold_vram_reserved_peak_mb":  sample_cold["vram_reserved_peak_mb"],
        "sample_cold_vram_reserved_mean_mb":  sample_cold["vram_reserved_mean_mb"],
        "sample_cold_vram_allocated_peak_mb": sample_cold["vram_allocated_peak_mb"],
        "sample_cold_vram_allocated_mean_mb": sample_cold["vram_allocated_mean_mb"],
        "sample_cold_ram_peak_mb":            sample_cold["ram_peak_mb"],
        "sample_cold_ram_mean_mb":            sample_cold["ram_mean_mb"],
        "sample_cold_cpu_util_peak":          sample_cold["cpu_util_peak"],
        "sample_cold_cpu_util_mean":          sample_cold["cpu_util_mean"],
        "sample_cold_gpu_util_peak":          sample_cold["gpu_util_peak"],
        "sample_cold_gpu_util_mean":          sample_cold["gpu_util_mean"],
        
        "sample_warm_vram_reserved_peak_mb":  sample_warm["vram_reserved_peak_mb"],
        "sample_warm_vram_reserved_mean_mb":  sample_warm["vram_reserved_mean_mb"],
        "sample_warm_vram_allocated_peak_mb": sample_warm["vram_allocated_peak_mb"],
        "sample_warm_vram_allocated_mean_mb": sample_warm["vram_allocated_mean_mb"],
        
        # Whole-run aggregates
        "ram_peak_mb":       m.ram_max_mb,
        "ram_mean_mb":       m.ram_mean_mb,
        "gpu_util_mean":     m.gpu_util_mean,
        "gpu_util_max":      m.gpu_util_max,
        "cpu_util_mean":     m.cpu_util_mean,
        "cpu_util_max":      m.cpu_util_max,
        "power_mean_w":      m.power_mean_watts  or 0,
        "power_max_w":       m.power_max_watts   or 0,
        "pcie_tx_mean_kb_s": m.pcie_tx_mean_kb_s or 0,
        "pcie_tx_max_kb_s":  m.pcie_tx_max_kb_s  or 0,
        "pcie_rx_mean_kb_s": m.pcie_rx_mean_kb_s or 0,
        "pcie_rx_max_kb_s":  m.pcie_rx_max_kb_s  or 0,
    }
    
    return result


# Timeline plot

def interpolate_series(time_arr: np.ndarray, values: np.ndarray,
                       common_t: np.ndarray) -> np.ndarray:
    return np.interp(common_t, time_arr, values,
                     left=values[0], right=values[-1])


def plot_averaged_timeline(run_dir: Path, run_results: list[dict],
                           title: str, save_path: Path):
    N_POINTS = 300
    ts_list, ev_list = [], []
    for i in range(len(run_results)):
        ts_f = run_dir / f"run_{i:02d}_timeseries.csv"
        ev_f = run_dir / f"run_{i:02d}_events.csv"
        if ts_f.exists():
            ts_list.append(pd.read_csv(ts_f))
        if ev_f.exists():
            ev_list.append(pd.read_csv(ev_f).set_index("event")["time_s"].to_dict())

    if not ts_list:
        logger.warning("No timeseries found in %s", run_dir)
        return

    time_col = "time" if "time" in ts_list[0].columns else "time_s"
    max_t    = float(np.median([df[time_col].iloc[-1] for df in ts_list]))
    common_t = np.linspace(0, max_t, N_POINTS)

    metrics_cfg = [
        ("vram",        "VRAM (MB)",           None,      None,     GAMER_VRAM_GB * 1024),
        ("ram_used_mb", "RAM / RSS (MB)",       "#3498db", None,     GAMER_RAM_GB  * 1024),
        ("gpu_util",    "GPU Utilisation (%)",  "#2ecc71", (0, 100), None),
        ("cpu_util",    "CPU Utilisation (%)",  "#f39c12", (0, 100), None),
        ("power_watts", "Power (W)",            "#9b59b6", None,     None),
        ("pcie",        "PCIe (MB/s)",          None,      None,     None),
    ]

    def avg_ev(key):
        vals = [ev[key] for ev in ev_list if key in ev]
        return float(np.mean(vals)) if vals else None

    def interp(col):
        rows = []
        for df in ts_list:
            if col not in df.columns:
                continue
            v = df[col].ffill().fillna(0).to_numpy(dtype=float)
            t = df[time_col].to_numpy(dtype=float)
            rows.append(interpolate_series(t, v, common_t))
        return np.stack(rows) if rows else None

    def phase_lines(ax):
        # Real phase markers from events
        ev_cfg = [
            # Cold run phases
            ("cold_start",              "COLD START",         "-"),
            ("text_encoder_load_start", "TE load",            ":"),
            ("text_encoder_load_end",   "TE loaded",          ":"),
            ("encode_start",            "Encode",             ":"),
            ("encode_end",              "Encoded",            ":"),
            ("diffusion_load_start",    "Diff load",          ":"),
            ("diffusion_load_end",      "Diff loaded",        ":"),
            ("sample_start",            "Sample",             ":"),
            ("sample_end",              "Sampled",            ":"),
            ("vae_load_start",          "VAE load",           ":"),
            ("vae_load_end",            "VAE loaded",         ":"),
            ("vae_decode_start",        "VAE decode",         ":"),
            ("vae_decode_end",          "Decoded",            ":"),
            ("cold_end",                "COLD END",           "--"),
            # Warm run phases
            ("warm_start",              "WARM START",         "-"),
            ("cleanup_start",           "Cleanup",            "-."),
            ("cleanup_end",             "END",                "-"),
        ]
        yl = ax.get_ylim()
        yr = yl[1] - yl[0]
        y_top = yl[1] - yr * 0.10
        y_bot = yl[0] + yr * 0.02
        top   = True
        for key, lbl, ls in ev_cfg:
            t = avg_ev(key)
            if t is None:
                continue
            ax.axvline(t, color="grey", linewidth=1.1, linestyle=ls, alpha=0.75)
            ax.text(t, y_top if top else y_bot, lbl, fontsize=7,
                    fontweight="bold", color="grey",
                    va="top" if top else "bottom",
                    ha="left", rotation=90, rotation_mode="anchor")
            top = not top

    fig, axes = plt.subplots(len(metrics_cfg), 1,
                             figsize=(14, 4 * len(metrics_cfg)), sharex=True)
    fig.suptitle(f"{title}\n{GAMER_LABEL}", fontsize=13, fontweight="bold", y=1.01)

    for ax, (col, ylabel, color, ylim_fixed, limit_mb) in zip(axes, metrics_cfg):
        if col == "vram":
            mat_r = interp("vram_reserved_mb")
            mat_a = interp("vram_allocated_mb")
            if mat_r is None:
                ax.set_visible(False); continue
            mr, sr = mat_r.mean(0), mat_r.std(0)
            ax.fill_between(common_t, mr - sr, mr + sr, alpha=0.15, color="#e74c3c")
            ax.plot(common_t, mr, lw=2.2, color="#e74c3c", label="reserved (mean ± std)")
            if mat_a is not None:
                ax.plot(common_t, mat_a.mean(0), lw=1.8, color="#e67e22",
                        ls="--", label="allocated")
            if limit_mb:
                ax.axhline(limit_mb, color="black", lw=1.5, ls=":",
                           alpha=0.85, label=f"gamer limit ({limit_mb/1024:.0f} GB)")
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(True, alpha=0.25, ls="--"); ax.set_axisbelow(True)
            ax.legend(fontsize=9, loc="upper right"); phase_lines(ax); continue

        if col == "pcie":
            mat_tx = interp("pcie_tx_kb_s")
            mat_rx = interp("pcie_rx_kb_s")
            if mat_tx is None and mat_rx is None:
                ax.set_visible(False); continue
            for mat, c, lbl in [
                (mat_tx, "#8e44ad", "TX (mean ± std)"),
                (mat_rx, "#2980b9", "RX (mean ± std)"),
            ]:
                if mat is None:
                    continue
                mean = mat.mean(0) / 1024
                std  = mat.std(0)  / 1024
                ax.fill_between(common_t, mean - std, mean + std, alpha=0.15, color=c)
                ax.plot(common_t, mean, lw=2.0, color=c,
                        ls="--" if "RX" in lbl else "-", label=lbl)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(True, alpha=0.25, ls="--"); ax.set_axisbelow(True)
            ax.legend(fontsize=9, loc="upper right"); phase_lines(ax); continue

        mat = interp(col)
        if mat is None:
            ax.set_visible(False); continue
        mean, std = mat.mean(0), mat.std(0)
        ax.fill_between(common_t, mean - std, mean + std, alpha=0.20, color=color, lw=0)
        ax.plot(common_t, mean, lw=2.2, color=color, label="mean ± 1 std")
        if limit_mb:
            ax.axhline(limit_mb, color="black", lw=1.5, ls=":",
                       alpha=0.85, label=f"gamer limit ({limit_mb/1024:.0f} GB)")
        ax.set_ylabel(ylabel, fontsize=12)
        if ylim_fixed:
            ax.set_ylim(*ylim_fixed)
        ax.grid(True, alpha=0.25, ls="--"); ax.set_axisbelow(True)
        ax.legend(fontsize=9, loc="upper right"); phase_lines(ax)

    axes[-1].set_xlabel("Time (s)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved: %s", save_path.name)


# Collect + plot

def _collect_and_plot(output_dir: Path) -> list[dict]:
    run_dir = output_dir / _RUN_ID
    results = []
    for i in range(N_RUNS):
        p = run_dir / f"scalar_{i:02d}.json"
        if p.exists():
            results.append(json.loads(p.read_text(encoding="utf-8")))
        else:
            logger.warning("Missing scalar for run %d", i)

    if not results:
        return []

    runs_df = pd.DataFrame(results)
    runs_df["model"]   = MODEL_CFG["name"]
    runs_df["offload"] = MODEL_CFG["offload"]
    runs_df["run_id"]  = _RUN_ID
    runs_df.to_csv(run_dir / "runs.csv", index=False)

    plot_averaged_timeline(
        run_dir, results,
        f"Resource Usage — {_RUN_ID}\n"
        f"(mean ± std over {len(results)} runs, fresh subprocess each)",
        save_path=output_dir / f"{_RUN_ID}_timeline.png",
    )
    return [dict(r, model=MODEL_CFG["name"],
                    offload=MODEL_CFG["offload"],
                    run_id=_RUN_ID)
            for r in results]


# Merge with existing all_runs.csv / summary.csv

def _merge_and_save(output_dir: Path, new_results: list[dict]):
    new_df        = pd.DataFrame(new_results)
    all_runs_path = output_dir / "all_runs.csv"

    if all_runs_path.exists():
        existing = pd.read_csv(all_runs_path)
        if "run_id" in existing.columns:
            existing = existing[existing["run_id"] != _RUN_ID]
        full_df = pd.concat([existing, new_df], ignore_index=True, sort=False)
    else:
        full_df = new_df

    full_df.to_csv(all_runs_path, index=False)
    logger.info("all_runs.csv updated (%d total rows)", len(full_df))

    numeric_cols = [c for c in full_df.columns
                    if c not in ("run", "seed", "model", "offload", "run_id")
                    and pd.api.types.is_numeric_dtype(full_df[c])]

    summary_rows = []
    for rid, grp in full_df.groupby("run_id"):
        row = {"run_id": rid, "model": grp["model"].iloc[0],
               "offload": grp["offload"].iloc[0], "n_runs": len(grp)}
        for col in numeric_cols:
            row[f"{col}_mean"] = grp[col].mean()
            row[f"{col}_std"]  = grp[col].std()
            if "vram_allocated" in col:
                std = row[f"{col}_std"]
                n   = len(grp)
                row[f"{col}_ci95"] = (
                    1.96 * std / math.sqrt(n)
                    if pd.notna(std) and n > 0 else float("nan")
                )
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)
    logger.info("summary.csv updated (%d configs)", len(summary_rows))


# Worker + orchestrator

def _worker_mode(run_index: int, output_dir: Path):
    """
    Runs inside a fresh subprocess.
    """
    _setup_comfyui_vram_and_paths()

    run_dir = output_dir / _RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = 42 + run_index

    logger.info("Worker starting: %s  run=%d  seed=%d", _RUN_ID, run_index, seed)
    r = single_run(run_index, seed, run_dir)

    logger.info(
        "TIMING SUMMARY:\n"
        "  Cold: TE_load=%.2fs encode=%.2fs diff_load=%.2fs sample=%.2fs vae=%.2fs total=%.2fs\n"
        "  Warm: TE_load=%.2fs encode=%.2fs diff_load=%.2fs sample=%.2fs vae=%.2fs total=%.2fs\n"
        "  VRAM_reserved_peak=%.0fMB  VRAM_allocated_peak=%.0fMB  pytorch_peak=%.0fMB",
        r["cold_text_encoder_load_time_s"], r["cold_encode_time_s"],
        r["cold_diffusion_load_time_s"], r["cold_sample_time_s"],
        r["cold_vae_load_time_s"] + r["cold_vae_decode_time_s"], r["cold_total_time_s"],
        r["warm_text_encoder_load_time_s"], r["warm_encode_time_s"],
        r["warm_diffusion_load_time_s"], r["warm_sample_time_s"],
        r["warm_vae_load_time_s"] + r["warm_vae_decode_time_s"], r["warm_total_time_s"],
        r["vram_reserved_peak_mb"],
        r["vram_allocated_peak_mb"],
        r["vram_allocated_pytorch_peak_mb"],
    )
    (run_dir / f"scalar_{run_index:02d}.json").write_text(
        json.dumps(r, default=str), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--_worker",  default=False, action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--_run",     default=None, type=int, dest="_run",
                        help=argparse.SUPPRESS)
    parser.add_argument("--_out-dir", default=None, dest="_out_dir",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._worker:
        _worker_mode(args._run, Path(args._out_dir))
        return

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COMFYUI OFFLOAD BENCHMARK  (--lowvram --reserve-vram) [FIXED]")
    print("With REAL timestamps for each phase")
    print("=" * 70)
    print(f"Config:       {_RUN_ID}")
    print(f"Runs:         {N_RUNS}  (one fresh subprocess per run)")
    print(f"ComfyUI dir:  {COMFYUI_ROOT}")
    print(f"VRAM cap:     {GAMER_VRAM_GB} GB  (--reserve-vram {RESERVE_VRAM_GB})")
    print(f"UNET:         {UNET_NAME}")
    print(f"CLIP:         {CLIP_NAME}")
    print(f"VAE:          {VAE_NAME}")
    print(f"Cold prompt:  {PROMPT[:70]}...")
    print(f"Warm prompt:  {WARM_PROMPT}")
    print(f"Output:       {output_dir}/")
    print("=" * 70)

    for run_index in range(N_RUNS):
        cmd = [
            COMFYUI_PYTHON, __file__,
            "--_worker",
            "--_run",     str(run_index),
            "--_out-dir", str(output_dir),
        ]
        print(f"\n  Spawning worker subprocess for run {run_index} ...", flush=True)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"  [X] Worker exited with code {result.returncode}")

    all_results = _collect_and_plot(output_dir)
    if not all_results:
        print("\nNo results collected.")
        return

    _merge_and_save(output_dir, all_results)

    df = pd.DataFrame(all_results)
    print("\n" + "=" * 70)
    print("COMFYUI BENCHMARK SUMMARY  (mean ± std)")
    print("=" * 70)
    print("\n--- TIMING (seconds) ---")
    for col in [
        "cold_text_encoder_load_time_s", "cold_encode_time_s",
        "cold_diffusion_load_time_s", "cold_sample_time_s",
        "cold_vae_load_time_s", "cold_vae_decode_time_s", "cold_total_time_s",
        "warm_text_encoder_load_time_s", "warm_encode_time_s",
        "warm_diffusion_load_time_s", "warm_sample_time_s",
        "warm_vae_load_time_s", "warm_vae_decode_time_s", "warm_total_time_s",
        "total_time_s",
    ]:
        if col in df.columns:
            print(f"  {col:40s}  {df[col].mean():.2f} ± {df[col].std():.2f}")
    
    print("\n--- VRAM (MB) ---")
    for col in [
        "vram_reserved_peak_mb", "vram_allocated_peak_mb",
        "vram_allocated_pytorch_peak_mb", "ram_peak_mb",
    ]:
        if col in df.columns:
            print(f"  {col:40s}  {df[col].mean():.2f} ± {df[col].std():.2f}")

    print(f"\nDone. Results in: {output_dir}/")


if __name__ == "__main__":
    main()
