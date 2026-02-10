"""Compare resource usage between models - simplified"""

import torch
import os
from src.models import load_pipeline
from src.monitoring import ResourceMonitor, cleanup_gpu


def benchmark_model(name, model_id, gguf_path=None):
    """Benchmark a single model"""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")
    
    # Load model
    pipe = load_pipeline(
        model_id=model_id,
        gguf_path=gguf_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16
    )
    
    # Run with monitoring
    prompt = "A serene lake surrounded by mountains at dawn"
    
    with ResourceMonitor() as monitor:
        image = pipe.generate(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=4,
            seed=42,
            output_path=f"benchmark_results/{name.replace(' ', '_')}.png"
        )
    
    metrics = monitor.get_metrics()
    metrics.print_summary()
    metrics.save_csv(f"benchmark_results/{name.replace(' ', '_')}.csv")
    metrics.plot(f"benchmark_results/{name.replace(' ', '_')}.png")
    
    cleanup_gpu()
    
    return {
        'model': name,
        'duration_s': metrics.duration_seconds,
        'vram_peak_mb': metrics.vram_max_mb,
        'vram_mean_mb': metrics.vram_mean_mb,
        'ram_peak_mb': metrics.ram_max_mb,
        'gpu_util_mean': metrics.gpu_util_mean,
        'power_mean_w': metrics.power_mean_watts,
        'energy_j': metrics.power_total_joules
    }


def main():
    os.makedirs("benchmark_results", exist_ok=True)
    
    # Models to compare
    configs = [
        {
            'name': 'Standard FP16',
            'model_id': 'black-forest-labs/FLUX.2-klein-4B',
            'gguf_path': None
        },
        {
            'name': 'GGUF Q4_K_M',
            'model_id': 'black-forest-labs/FLUX.2-klein-4B',
            'gguf_path': 'unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q4_K_M.gguf'
        },
        {
            'name': 'GGUF Q2_K',
            'model_id': 'black-forest-labs/FLUX.2-klein-4B',
            'gguf_path': 'unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q2_K.gguf'
        }
    ]
    
    # Benchmark each
    results = []
    for config in configs:
        try:
            result = benchmark_model(**config)
            results.append(result)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Print comparison table
    if results:
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"{'Model':<20} {'Time(s)':<10} {'VRAM(MB)':<12} {'RAM(MB)':<12} {'GPU%':<8}")
        print("-"*80)
        for r in results:
            print(f"{r['model']:<20} {r['duration_s']:<10.2f} "
                  f"{r['vram_peak_mb']:<12.0f} {r['ram_peak_mb']:<12.0f} "
                  f"{r['gpu_util_mean']:<8.1f}")
        print(f"{'='*80}\n")
        
        # Create comparison plot
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
            
            models = [r['model'] for r in results]
            x_pos = np.arange(len(models))
            
            # Time
            axes[0, 0].bar(x_pos, [r['duration_s'] for r in results], color='blue', alpha=0.7)
            axes[0, 0].set_ylabel('Time (s)')
            axes[0, 0].set_title('Generation Time')
            axes[0, 0].set_xticks(x_pos)
            axes[0, 0].set_xticklabels(models, rotation=15, ha='right')
            
            # VRAM
            axes[0, 1].bar(x_pos, [r['vram_peak_mb'] for r in results], color='red', alpha=0.7)
            axes[0, 1].set_ylabel('VRAM (MB)')
            axes[0, 1].set_title('Peak VRAM')
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels(models, rotation=15, ha='right')
            
            # RAM
            axes[1, 0].bar(x_pos, [r['ram_peak_mb'] for r in results], color='green', alpha=0.7)
            axes[1, 0].set_ylabel('RAM (MB)')
            axes[1, 0].set_title('Peak RAM')
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(models, rotation=15, ha='right')
            
            # GPU Util
            axes[1, 1].bar(x_pos, [r['gpu_util_mean'] for r in results], color='orange', alpha=0.7)
            axes[1, 1].set_ylabel('GPU Utilization (%)')
            axes[1, 1].set_title('Average GPU Usage')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(models, rotation=15, ha='right')
            
            plt.tight_layout()
            plt.savefig("benchmark_results/comparison.png", dpi=150, bbox_inches='tight')
            print("✓ Saved comparison chart")
            plt.close()
        except Exception as e:
            print(f"⚠ Could not create plots: {e}")


if __name__ == "__main__":
    main()
