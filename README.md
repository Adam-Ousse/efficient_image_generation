# Efficient Inference Benchmarking (Flux.2)

This repository contains the efficient inference part of the CSC_5IA21 project:
https://giannifranchi.github.io/CSC_5IA21.html

The code benchmarks FLUX.2 variants under constrained hardware, with emphasis on quantization, offloading, monitoring, and OCR-aware quality evaluation.

![Peak VRAM FP16](figures/benchmark/fig_vram_fp16.png)
![Quality report](figures/qualitative/quality_report.png)
## Project Scope

- Focus: inference efficiency only (not full training/fine-tuning).
- Main question: what setup is best for frugal hardware (low VRAM/RAM).
- Comparison axis: latency, memory, system load, and output quality.

## Models and Strategies

Benchmarked model family:
- FLUX.2-Klein FP16 baseline.
- GGUF quantized variants: Q2_K, Q3_K_M, Q4_K_M, Q5_K_M.

Benchmarked execution strategies:
- Full GPU.
- CPU offload / sequential / group offload.
- CPU-only baseline.
- Custom smart offloading.
- ComfyUI low-VRAM path.

## What We Track

- Latency: load, encode, generate, total.
- Memory: VRAM reserved/allocated, RAM RSS.
- Transfers: PCIe TX/RX.
- Utilization and energy: GPU/CPU usage, power, energy per image.
- Text/quality metrics: OCR (CER, normalized CER, WER), SSIM, FID, CLIP.

## Repository Structure

```text
efficient_image_generation/
  src/
    models/        # model descriptors/loaders
    offload/       # smart offloading + pipeline patches
    monitoring/    # runtime monitor + metrics aggregation
    evaluation/    # OCR, SSIM, FID, CLIP modules
  benchmark_offload.py        # hardware/offloading benchmark orchestration
  benchmark_models.py         # prompt/seed/model image generation benchmark
  comfy_benchmark.py          # ComfyUI low-VRAM benchmark path
  evaluate_ocr.py             # OCR evaluation pipeline
  evaluate_ssim_clip_fid.py   # SSIM/FID/CLIP evaluation pipeline
  report_figures.py           # report figure generation
  figures/                    # benchmark and qualitative figures
  results/                    # generated images and CSV outputs
```

## Core Scripts and Code Paths

### 1) Hardware + offloading benchmark

Main orchestration: `benchmark_offload.py`

- Runs full lifecycle phases per run: load -> encode cold -> gen cold -> encode warm -> gen warm -> cleanup.
- Saves per-run traces and aggregated summaries.

Example:

```bash
python benchmark_offload.py
```

### 2) Smart offloading implementation

Main source: `src/offload/offload.py`

- `SmartOffloadManager` classifies modules into resident vs streaming according to VRAM budget.
- Uses pinned CPU memory + CUDA streams + hook-based tensor swapping.
- Includes GGUF-specific dequant ring-buffer logic.

Minimal usage pattern:

```python
from src.offload.offload import SmartOffloadManager

mgr = SmartOffloadManager(transformer, max_vram_gb=6.0, device="cuda", num_streams=2)
mgr.load()
# run inference
mgr.unload()
```

### 3) Monitoring implementation

Main sources:
- `src/monitoring/resource_monitor.py`
- `src/monitoring/metrics.py`

Sampling backend combines:
- `psutil` for process CPU/RAM,
- `pynvml` for GPU utilization/power/PCIe,
- torch CUDA allocator stats for allocated/reserved VRAM.

Minimal usage pattern:

```python
from src.monitoring import ResourceMonitor

with ResourceMonitor(sample_rate_hz=5.0) as mon:
    # run generation
    pass

metrics = mon.get_metrics()
df = metrics.to_dataframe()
print(metrics.vram_reserved_max_mb, metrics.ram_max_mb)
```

### 4) OCR and quality evaluation

Main sources:
- `src/evaluation/ocr.py` and `evaluate_ocr.py`
- `src/evaluation/ssim.py`
- `src/evaluation/fid.py`
- `src/evaluation/clip_score.py`
- `evaluate_ssim_clip_fid.py`

OCR backends in code:
- EasyOCR.
- GLM-OCR (`zai-org/GLM-OCR`).

Example OCR run:

```bash
python evaluate_ocr.py
```

Example SSIM/FID/CLIP run:

```bash
python evaluate_ssim_clip_fid.py
```

## Benchmark Overview Figures

### Hardware monitoring pipeline
![Hardware benchmark overview](figures/hardware_benchmark_overview.png)

### OCR and evaluation pipeline
![OCR benchmark overview](figures/model_benchmark_ocr_overview.png)

## Benchmark Figures (Offloading + Hardware)


![Latency FP16](figures/benchmark/fig_timing_fp16.png)
![PCIe FP16](figures/benchmark/fig_pcie_fp16.png)
![Quantization impact](figures/benchmark/fig_quant.png)
![System RAM](figures/benchmark/fig_ram.png)
![Power FP16](figures/benchmark/fig_power_fp16.png)
![Power vs quantization](figures/benchmark/fig_power_quant.png)
![CPU offload timeline](figures/benchmark/FLUX2-Klein-FP16__CPUOff_timeline.png)

## Qualitative and Quality Figures

<summary>Show qualitative and quality report figures</summary>

![Comparison grid](figures/qualitative/comparison_grid.png)
![Distillation grid](figures/qualitative/grid_distill_6cols.jpg)
![Flux vs text rows](figures/qualitative/grid_flux_vs_text_rows_fixed.jpg)
![OCR report](figures/qualitative/ocr_report.png)


