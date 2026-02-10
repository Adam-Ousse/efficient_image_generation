# Efficient Image Generation Benchmarking

A clean, modular framework for benchmarking image generation models with support for both standard Diffusers and GGUF quantized models.

## Quick Start

### Installation

```bash
cd efficient_image_generation
pip install -r requirements.txt
```

### Basic Usage

```python
from src.models import load_pipeline

# Load standard FP16 model
pipe = load_pipeline(
    model_id="black-forest-labs/FLUX.2-klein-4B",
    device="cuda",
    dtype=torch.float16
)

# Generate image
image = pipe.generate(
    prompt="A cat in a hat",
    seed=42,
    output_path="output.png"
)
```

### GGUF Quantized Models

```python
# Load GGUF quantized model for lower memory usage
# Auto-downloads from HuggingFace if not found locally
pipe = load_pipeline(
    model_id="black-forest-labs/FLUX.2-klein-4B",
    gguf_path="unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q4_K_M.gguf",  # HF format
    device="cuda"
)

# Or use a local path
pipe = load_pipeline(
    model_id="black-forest-labs/FLUX.2-klein-4B",
    gguf_path="/path/to/flux-2-klein-4b-Q2_K.gguf",  # Local path
    device="cuda"
)

image = pipe.generate("A penguin on ice", seed=42)
```

### Resource Monitoring

Monitor VRAM, RAM, CPU/GPU utilization, and power during generation:

```python
from src.models import load_pipeline
from src.monitoring import ResourceMonitor

pipe = load_pipeline(
    model_id="black-forest-labs/FLUX.2-klein-4B",
    gguf_path="unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q2_K.gguf"
)

# Monitor during generation
with ResourceMonitor() as monitor:
    image = pipe.generate("A beautiful landscape", seed=42)

# Get metrics
metrics = monitor.get_metrics()
metrics.print_summary()      # Print stats
metrics.save_csv("data.csv") # Export CSV
metrics.plot("plot.png")     # Create plot

# Access DataFrame
df = metrics.to_dataframe()
print(f"Peak VRAM: {df['vram_used_mb'].max()} MB")
```

Tracks: VRAM, RAM, GPU/CPU utilization, power consumption. Returns pandas DataFrame for analysis.

```
