"""
benchmark_models.py — Image quality benchmark for FLUX GGUF quantisations.

Generates images for every (model, prompt, seed) combination and records
generation time. No hardware monitoring.

Structure
---------
  results/benchmark_<timestamp>/
    <prompt_label>/
      seed_<seed>/
        FLUX2-Klein-FP16.png
        FLUX2-Klein-Q5_K_M.png
        ...
      prompt.txt
    benchmark_results.csv

Usage
-----
    python benchmark_models.py
    python benchmark_models.py --fast   # first prompt × first seed only
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import traceback
import warnings
from pathlib import Path

import pandas as pd
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from src.models.flux import FluxModel
from src.models.zimage import ZImageModel
from prompts.prompts_benchmark_models import PROMPTS


_4B   = "black-forest-labs/FLUX.2-klein-4B"
_9B   = "black-forest-labs/FLUX.2-klein-9B"
_dev = "black-forest-labs/FLUX.2-dev"
_GGUF_4B = "unsloth/FLUX.2-klein-4B-GGUF"
_GGUF_9B = "unsloth/FLUX.2-klein-9B-GGUF"

MODELS = [
    {"name": "FLUX2-Klein-dev-FP16",   "model_id": _dev, "gguf_path": None},
    # {"name": "FLUX2-Klein-9B-FP16",   "model_id": _9B, "gguf_path": None},
    # {"name": "FLUX2-Klein-4B-FP16",   "model_id": _4B, "gguf_path": None},
    # {"name": "FLUX2-Klein-4B-Q5_K_M", "model_id": _4B, "gguf_path": f"{_GGUF_4B}/flux-2-klein-4b-Q5_K_M.gguf"},
    # {"name": "FLUX2-Klein-4B-Q4_K_M", "model_id": _4B, "gguf_path": f"{_GGUF_4B}/flux-2-klein-4b-Q4_K_M.gguf"},
    # {"name": "FLUX2-Klein-4B-Q3_K_M", "model_id": _4B, "gguf_path": f"{_GGUF_4B}/flux-2-klein-4b-Q3_K_M.gguf"},
    # {"name": "FLUX2-Klein-4B-Q2_K",   "model_id": _4B, "gguf_path": f"{_GGUF_4B}/flux-2-klein-4b-Q2_K.gguf"},
    # {"name": "FLUX2-Klein-9B-Q2_K",   "model_id": _9B, "gguf_path": f"{_GGUF_9B}/flux-2-klein-9b-Q2_K.gguf"},
    # {"name": "Z-Image",       "model_id": "Tongyi-MAI/Z-Image",       "gguf_path": None, "cls": ZImageModel},
    # {"name": "Z-Image-Turbo", "model_id": "Tongyi-MAI/Z-Image-Turbo", "gguf_path": None, "cls": ZImageModel},
]

SEEDS      = [42, 123, 456]
DEVICE     = "cuda"
DTYPE      = torch.bfloat16
GEN_PARAMS = {
    "height": 1024,
    "width":  1024,
    "num_inference_steps": 4,
    "guidance_scale": 1.0,
}



def _outpath(base: Path, prompt_label: str, seed: int, model_name: str) -> Path:
    p = base / prompt_label / f"seed_{seed}"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{model_name}.png"


def _save_prompt(base: Path, prompt_label: str, prompt_text: str):
    f = base / prompt_label / "prompt.txt"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(prompt_text, encoding="utf-8")



def benchmark_model(model_cfg: dict, prompts: list, seeds: list,
                    out_dir: Path) -> list[dict]:
    name = model_cfg["name"]

    print(f"\n{'═'*70}")
    print(f"  MODEL: {name}")
    print(f"{'═'*70}")

    model_cls   = model_cfg.get("cls", FluxModel)
    descriptor  = model_cls(model_cfg["model_id"], gguf_path=model_cfg["gguf_path"], dtype=DTYPE) \
                  if model_cls is FluxModel \
                  else model_cls(model_cfg["model_id"], dtype=DTYPE)
    t_load      = time.perf_counter()
    if model_cls is FluxModel:
        transformer = descriptor.load_transformer()
        pipe        = descriptor.load_pipeline(transformer).to(DEVICE)
    else:
        transformer = None
        pipe        = descriptor.load_pipeline().to(DEVICE)
    torch.cuda.synchronize()
    load_time   = time.perf_counter() - t_load
    print(f"  Load time : {load_time:.1f} s")

    rows  = []
    total = len(prompts) * len(seeds)
    idx   = 0

    for prompt_data in prompts:
        _save_prompt(out_dir, prompt_data["label"], prompt_data["prompt"])
        for seed in seeds:
            idx += 1
            out_path = _outpath(out_dir, prompt_data["label"], seed, name)
            print(f"  [{idx:3d}/{total}] {prompt_data['label']}  seed={seed}  ",
                  end="", flush=True)
            try:
                gen = torch.Generator(device=DEVICE).manual_seed(seed)
                t0  = time.perf_counter()
                out = pipe(prompt=prompt_data["prompt"], generator=gen, **GEN_PARAMS)
                torch.cuda.synchronize()
                gen_time = time.perf_counter() - t0
                out.images[0].save(out_path)
                print(f"{gen_time:.1f}s")
                rows.append({
                    "model":             name,
                    "prompt_label":      prompt_data["label"],
                    "seed":              seed,
                    "image_path":        str(out_path),
                    "load_time_s":       load_time,
                    "generation_time_s": gen_time,
                })
            except Exception as e:
                print(f"ERROR: {e}")
                traceback.print_exc()

    del pipe
    if transformer is not None:
        del transformer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return rows



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Smoke-test: first prompt × first seed only")
    args = parser.parse_args()

    prompts = PROMPTS[:1] if args.fast else PROMPTS
    seeds   = SEEDS[:1]   if args.fast else SEEDS

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # out_dir   = Path("results") / f"benchmark_{timestamp}"
    out_dir = Path("results") / "benchmark_latest"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("═" * 70)
    print("  FLUX QUANTISATION BENCHMARK")
    print("═" * 70)
    print(f"  Models  : {[m['name'] for m in MODELS]}")
    print(f"  Prompts : {len(prompts)}")
    print(f"  Seeds   : {seeds}")
    print(f"  Total   : {len(MODELS) * len(prompts) * len(seeds)} images")
    print(f"  Output  : {out_dir}")
    print("═" * 70)

    all_rows = []
    for model_cfg in MODELS:
        rows = benchmark_model(model_cfg, prompts, seeds, out_dir)
        all_rows.extend(rows)

    if all_rows:
        df = pd.DataFrame(all_rows)
        csv_path = out_dir / "benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n  Results → {csv_path}")

        print("\n" + "═" * 70)
        print("  SUMMARY  (mean generation time per model)")
        print("═" * 70)
        for model_cfg in MODELS:
            name = model_cfg["name"]
            sub  = df[(df["model"] == name) & (df["generation_time_s"] > 0)]
            if sub.empty:
                continue
            print(f"  {name:30s}  load {sub['load_time_s'].iloc[0]:.1f}s  "
                  f"gen {sub['generation_time_s'].mean():.1f} ± "
                  f"{sub['generation_time_s'].std():.1f}s")
        print("═" * 70)

    print(f"\n✓ Done. All results in {out_dir}/")


if __name__ == "__main__":
    main()
