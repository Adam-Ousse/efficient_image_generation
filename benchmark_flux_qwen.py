from __future__ import annotations

import argparse
import gc
import sys
import time
import traceback
import warnings
from itertools import product
from pathlib import Path

import pandas as pd
import torch

# Some installations of gguf report version 'N/A' (or omit __version__), which can crash transformers.
try:
    import gguf
    try:
        # packaging.version is a submodule, not an attribute of packaging.
        from packaging import version as packaging_version
    except Exception:
        packaging_version = None

    current_version = getattr(gguf, "__version__", None)
    if (not isinstance(current_version, str)) or (not current_version.strip()) or (current_version.strip().upper() == "N/A"):
        gguf.__version__ = "0.18.0"
    elif packaging_version is not None:
        try:
            packaging_version.parse(current_version)
        except Exception:
            gguf.__version__ = "0.18.0"
except ImportError:
    pass

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from src.models.flux import FluxModel
from prompts.prompts_benchmark_models import PROMPTS


FLUX_BASE_ID = "black-forest-labs/FLUX.2-klein-4B"
QWEN_BASE_ID = "Qwen/Qwen3-4B" 
QWEN_GGUF_ID = "unsloth/Qwen3-4B-GGUF"

TRANSFORMER_CONFIGS = [
    {"name": "FLUX-4B-FP16", "gguf_path": None},
    {"name": "FLUX-4B-Q5_K_M", "gguf_path": "unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q5_K_M.gguf"},
    {"name": "FLUX-4B-Q4_K_M", "gguf_path": "unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q4_K_M.gguf"},
    {"name": "FLUX-4B-Q3_K_M", "gguf_path": "unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q3_K_M.gguf"},
    {"name": "FLUX-4B-Q2_K",   "gguf_path": "unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q2_K.gguf"},
]

TEXT_ENCODER_CONFIGS = [
    {"name": "Qwen-4B-BF16", "gguf_path": None},
    # {"name": "Qwen-4B-Q8_0",   "gguf_path": f"{QWEN_GGUF_ID}/Qwen3-4B-Q8_0.gguf"},
    # {"name": "Qwen-4B-Q6_K",   "gguf_path": f"{QWEN_GGUF_ID}/Qwen3-4B-Q6_K.gguf"},
    {"name": "Qwen-4B-Q5_K_M", "gguf_path": f"{QWEN_GGUF_ID}/Qwen3-4B-Q5_K_M.gguf"},
    {"name": "Qwen-4B-Q4_K_M", "gguf_path": f"{QWEN_GGUF_ID}/Qwen3-4B-Q4_K_M.gguf"},
    {"name": "Qwen-4B-Q3_K_M", "gguf_path": f"{QWEN_GGUF_ID}/Qwen3-4B-Q3_K_M.gguf"},
    {"name": "Qwen-4B-Q2_K",   "gguf_path": f"{QWEN_GGUF_ID}/Qwen3-4B-Q2_K.gguf"},
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


def load_qwen_encoder(cfg: dict):
    """
    loads Qwen encoder andtokenizer 
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gguf_path = cfg["gguf_path"]
    
    tokenizer = AutoTokenizer.from_pretrained(QWEN_BASE_ID)
    
    if gguf_path:
        # GGUF Path format: "unsloth/Qwen3-4B-GGUF/Qwen3-4B-Q8_0.gguf"
        parts = gguf_path.split("/")
        if len(parts) < 3:
            raise ValueError(f"Invalid GGUF path format: {gguf_path}")
            
        repo_id = f"{parts[0]}/{parts[1]}"
        filename = "/".join(parts[2:])
        
        print(f"    Loading GGUF Text Encoder: {repo_id} / {filename}")
    
        encoder = AutoModelForCausalLM.from_pretrained(
            repo_id,
            gguf_file=filename,
            torch_dtype=DTYPE,
            device_map="auto"
        )
    else:
        # Standard FP16 load
        print(f"    Loading FP16 Text Encoder: {QWEN_BASE_ID}")
        encoder = AutoModelForCausalLM.from_pretrained(QWEN_BASE_ID, torch_dtype=DTYPE)
        
    return encoder, tokenizer

def load_flux_transformer(cfg: dict):
    gguf = cfg["gguf_path"]
    descriptor = FluxModel(FLUX_BASE_ID, gguf_path=gguf, dtype=DTYPE)
    transformer = descriptor.load_transformer()
    
    return transformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Smoke-test: first prompt × first seed only")
    args = parser.parse_args()

    prompts = PROMPTS[10:11] if args.fast else PROMPTS
    seeds   = SEEDS[:1]   if args.fast else SEEDS

    out_dir = Path("results") / "benchmark_flux_qwen_latest"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("═" * 70)
    print("  FLUX + QWEN BENCHMARK")
    print("═" * 70)
    print(f"  Output  : {out_dir}")
    print("═" * 70)

    all_rows = []
    
    combos = list(product(TRANSFORMER_CONFIGS, TEXT_ENCODER_CONFIGS))
    
    for trans_cfg, te_cfg in combos:
        combo_name = f"{trans_cfg['name']}__{te_cfg['name']}"
        
        print(f"\n{'═'*70}")
        print(f"  MODEL: {combo_name}")
        print(f"{'═'*70}")

        try:
            t_load_start = time.perf_counter()

            transformer = load_flux_transformer(trans_cfg).to(DEVICE)

            text_encoder, tokenizer = load_qwen_encoder(te_cfg)
            
            if hasattr(text_encoder, "to"):
                text_encoder = text_encoder.to(DEVICE)

    
            from diffusers import Flux2KleinPipeline
            pipe = Flux2KleinPipeline.from_pretrained(
                FLUX_BASE_ID,
                transformer=transformer,
                text_encoder=text_encoder,
                tokenizer=tokenizer,  
                torch_dtype=DTYPE,
            ).to(DEVICE)
            
            torch.cuda.synchronize()
            load_time = time.perf_counter() - t_load_start
            print(f"  Load time : {load_time:.1f} s")

            total = len(prompts) * len(seeds)
            idx = 0
            
            for prompt_data in prompts:
                _save_prompt(out_dir, prompt_data["label"], prompt_data["prompt"])
                for seed in seeds:
                    idx += 1
                    out_path = _outpath(out_dir, prompt_data["label"], seed, combo_name)
                    
                    print(f"  [{idx:3d}/{total}] {prompt_data['label']}  seed={seed}  ", end="", flush=True)
                    
                    try:
                        gen = torch.Generator(device=DEVICE).manual_seed(seed)
                        t0  = time.perf_counter()
                        
                        out = pipe(prompt=prompt_data["prompt"], generator=gen, **GEN_PARAMS)
                        
                        torch.cuda.synchronize()
                        gen_time = time.perf_counter() - t0
                        out.images[0].save(out_path)
                        print(f"{gen_time:.1f}s")
                        
                        all_rows.append({
                            "model":             combo_name,
                            "prompt_label":      prompt_data["label"],
                            "seed":              seed,
                            "image_path":        str(out_path),
                            "load_time_s":       load_time,
                            "generation_time_s": gen_time,
                        })
                    except Exception as e:
                        print(f"ERROR: {e}")
                        traceback.print_exc()

        except Exception as e:
            print(f"  ERROR loading model combo {combo_name}: {e}")
            traceback.print_exc()

        finally:
            if 'pipe' in locals(): del pipe
            if 'transformer' in locals(): del transformer
            if 'text_encoder' in locals(): del text_encoder
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    if all_rows:
        df = pd.DataFrame(all_rows)
        csv_path = out_dir / "benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n  Results → {csv_path}")

        print("\n" + "═" * 70)
        print("  SUMMARY")
        print("═" * 70)
        for name in df["model"].unique():
            sub = df[df["model"] == name]
            if sub.empty: continue
            print(f"  {name:40s}  load {sub['load_time_s'].iloc[0]:.1f}s  "
                  f"gen {sub['generation_time_s'].mean():.1f} ± {sub['generation_time_s'].std():.1f}s")
        print("═" * 70)

    print(f"\n✓ Done. All results in {out_dir}/")

if __name__ == "__main__":
    main()