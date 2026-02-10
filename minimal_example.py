import torch
from src.models import load_pipeline
from src.monitoring import ResourceMonitor

def main():
    print("Minimal Resource Monitoring Example\n")
    
    # 1. Load model (auto-downloads GGUF if needed)
    print("Loading model...")
    
    pipe = load_pipeline(
        model_id="black-forest-labs/FLUX.2-klein-4B",
        gguf_path = "unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q2_K.gguf",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16
        )
    # 2. Generate with monitoring
    print("\nGenerating image with resource monitoring...")
    with ResourceMonitor(sample_rate_hz=60.0) as monitor:
        
        image = pipe.generate(
            prompt="A peaceful zen garden with cherry blossoms",
            height=1024,
            width=1024,
            num_inference_steps=4,
            seed=42,
            output_path="minimal_output.png"
        )
    
    # 3. Get and display results
    metrics = monitor.get_metrics()
    metrics.print_summary()
    
    # 4. Save results
    metrics.save_csv("minimal_metrics.csv")
    metrics.plot("minimal_plot.png")
    
    print("\n✓ Done!")
    print("  Image: minimal_output.png")
    print("  Metrics: minimal_metrics.csv")
    print("  Plot: minimal_plot.png")

if __name__ == "__main__":
    main()
