"""
Benchmark multiple models across prompts and seeds
Generates images and collects resource metrics
"""

import torch
import time
import pandas as pd
from pathlib import Path
from src.models import load_pipeline
from src.monitoring import ResourceMonitor, cleanup_gpu


# ============================================================================
# CONFIGURATION - Edit these to customize your benchmark
# ============================================================================

# Models to benchmark
MODELS = [
    {
        "name": "FLUX2-Klein-FP16",
        "model_id": "black-forest-labs/FLUX.2-klein-4B",
        "gguf_path": None
    },
    {
        "name": "FLUX2-Klein-Q4_K_M",
        "model_id": "black-forest-labs/FLUX.2-klein-4B",
        "gguf_path": "unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q4_K_M.gguf"
    },
    {
        "name": "FLUX2-Klein-Q2_K",
        "model_id": "black-forest-labs/FLUX.2-klein-4B",
        "gguf_path": "unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q2_K.gguf"
    }
]

# Prompts to test
PROMPTS = [
    {"label": "Human Portrait", "prompt": "A photorealistic portrait of a man in his 50s with short graying hair and slight wrinkles, sitting at a small table on a busy street, smoking a cigarette and drinking an espresso, wearing a casual jacket and shirt, warm golden-hour light illuminating his face, candid style, shallow depth of field with softly blurred background of pedestrians and street cafés."},
    {"label": "Street Scene", "prompt": "A deserted city street at dusk with long shadows, wet pavement reflecting neon lights, pedestrians in motion blur, cinematic mood, high detail."},
    {"label": "Classroom with Text", "prompt": "A classroom scene with a large blackboard showing the text 'Laws of motion :\n Newton's Second Law: F = m x a' written clearly in white chalk, students at desks taking notes, soft natural light streaming through the windows, educational infographic style, neatly organized desks and chairs, visible classroom details like books and a globe on a side table."},
    {"label": "Jungle", "prompt": "A dense jungle clearing with tall trees, thick vines, shafts of sunlight through the canopy, a distant waterfall cascading in the background, ultra-detailed foliage."},
    {"label": "Cat", "prompt": "A curious cat sitting on a stone wall in a garden, soft morning light, detailed fur texture and expressive eyes, shallow depth of field macro feel."},
    {"label": "Architecture", "prompt": "Modern architectural exterior of a minimalist building with clean concrete and glass surfaces, wide-angle perspective, crisp shadows, landscaped foreground and open plaza."},
    {"label": "Website Mockup", "prompt": "Website mockup of a clean landing page showing a header, navigation bar, main hero section with a headline and call-to-action button, three feature cards with descriptive text and icons."},
    {"label": "Factory Scene with Object Positioning", "prompt": "A large industrial factory interior with a metal conveyor belt running through the center of the scene, a bright red toolbox placed to the left of the conveyor belt, a forklift parked near the back wall on the right, stacks of wooden crates along the far-left wall, and overhead fluorescent lights casting clear shadows across the polished concrete floor. Realistic machinery details and industrial textures are visible."},
    {"label": "Macro Shot with Reflections", "prompt": "Macro close-up of a polished gemstone with intricate facets reflecting light, sharp focus on surface detail, soft bokeh background, high contrast reflections."},
    {"label": "Mirror with Subject Positioning", "prompt": "A photorealistic portrait of a man in his 40s with short brown hair and a trimmed beard, wearing a navy sweater and jeans, standing slightly to the right in front of a large wall-mounted mirror, so that both his front and profile are visible in the reflection, soft indoor lighting from a nearby lamp casting gentle shadows, a minimalist living room background with a wooden floor and a potted plant."}
]

# Seeds
SEEDS = [42, 123, 456]

# Generation parameters
GEN_PARAMS = {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 4,
    "guidance_scale": 1.0
}

# ============================================================================


def generate_image(pipe, prompt_data, seed, model_name, base_output_dir):
    """Generate a single image with monitoring"""
    
    # Create prompt-centric folder structure: prompt/seed_XX/model.png
    prompt_dir = base_output_dir / prompt_data['label'].replace(' ', '_')
    seed_dir = prompt_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{model_name}.png"
    filepath = seed_dir / filename
    
    print(f"\nGenerating: {prompt_data['label']} (seed {seed})")
    
    # Monitor during generation
    with ResourceMonitor(sample_rate_hz=60.0) as monitor:
        start_time = time.time()
        
        image = pipe.generate(
            prompt=prompt_data['prompt'],
            seed=seed,
            output_path=str(filepath),
            **GEN_PARAMS
        )
        
        generation_time = time.time() - start_time
    
    # Get metrics
    metrics = monitor.get_metrics()
    
    # Return results
    return {
        'model': model_name,
        'prompt_label': prompt_data['label'],
        'seed': seed,
        'generation_time_s': generation_time,
        'vram_peak_mb': metrics.vram_max_mb,
        'vram_mean_mb': metrics.vram_mean_mb,
        'ram_peak_mb': metrics.ram_max_mb,
        'ram_mean_mb': metrics.ram_mean_mb,
        'gpu_util_mean': metrics.gpu_util_mean,
        'gpu_util_max': metrics.gpu_util_max,
        'cpu_util_mean': metrics.cpu_util_mean,
        'cpu_util_max': metrics.cpu_util_max,
        'power_mean_watts': metrics.power_mean_watts if metrics.power_mean_watts else 0,
        'power_max_watts': metrics.power_max_watts if metrics.power_max_watts else 0,
        'energy_total_joules': metrics.power_total_joules if metrics.power_total_joules else 0,
        'image_path': str(filepath)
    }


def benchmark_model(model_config, prompts, seeds, base_output_dir):
    """Benchmark one model across all prompts and seeds"""
    
    model_name = model_config['name']
    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {model_name}")
    print(f"{'='*80}")
    
    # Load model
    print(f"\nLoading {model_name}...")
    pipe = load_pipeline(
        model_id=model_config['model_id'],
        gguf_path=model_config['gguf_path'],
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16,
        enable_cpu_offload=False,
        enable_vae_slicing=True
    )
    
    # Collect results
    results = []
    total = len(prompts) * len(seeds)
    current = 0
    
    # Generate images
    for prompt_data in prompts:
        for seed in seeds:
            current += 1
            print(f"\n[{current}/{total}] ", end="")
            
            try:
                result = generate_image(pipe, prompt_data, seed, model_name, base_output_dir)
                results.append(result)
                
                # Print quick summary
                print(f"  ✓ Time: {result['generation_time_s']:.2f}s | "
                      f"VRAM: {result['vram_peak_mb']:.0f}MB | "
                      f"GPU: {result['gpu_util_mean']:.1f}%")
                
            except Exception as e:
                print(f"  ✗ ERROR: {e}")
                # Still record the failure
                results.append({
                    'model': model_name,
                    'prompt_label': prompt_data['label'],
                    'seed': seed,
                    'generation_time_s': 0,
                    'vram_peak_mb': 0,
                    'vram_mean_mb': 0,
                    'ram_peak_mb': 0,
                    'ram_mean_mb': 0,
                    'gpu_util_mean': 0,
                    'gpu_util_max': 0,
                    'cpu_util_mean': 0,
                    'cpu_util_max': 0,
                    'power_mean_watts': 0,
                    'power_max_watts': 0,
                    'energy_total_joules': 0,
                    'image_path': 'ERROR',
                    'error': str(e)
                })
    
    # Cleanup - properly unload model
    print(f"\nCleaning up {model_name}...")
    del pipe
    cleanup_gpu()
    
    # Wait for cleanup to complete
    print("Waiting for GPU cleanup...")
    time.sleep(3)
    
    return results


def main():
    """Run complete benchmark"""
    
    # Setup output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"benchmark_results_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("IMAGE GENERATION BENCHMARK")
    print("="*80)
    print(f"\nModels: {len(MODELS)}")
    print(f"Prompts: {len(PROMPTS)}")
    print(f"Seeds: {len(SEEDS)}")
    print(f"Total generations: {len(MODELS) * len(PROMPTS) * len(SEEDS)}")
    print(f"\nOutput directory: {output_dir}")
    print("="*80)
    
    # Benchmark each model
    all_results = []
    
    for idx, model_config in enumerate(MODELS):
        try:
            model_results = benchmark_model(model_config, PROMPTS, SEEDS, output_dir)
            all_results.extend(model_results)
            
            # Sleep between models to ensure clean separation
            if idx < len(MODELS) - 1:  # Don't sleep after last model
                print(f"\nWaiting before next model...")
                time.sleep(5)
                
        except Exception as e:
            print(f"\n❌ ERROR benchmarking {model_config['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save results
    csv_path = output_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved full results to: {csv_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    # Summary by model
    summary = df.groupby('model').agg({
        'generation_time_s': ['mean', 'std'],
        'vram_peak_mb': ['mean', 'max'],
        'gpu_util_mean': 'mean',
        'cpu_util_mean': 'mean',
        'power_mean_watts': 'mean',
        'energy_total_joules': 'sum'
    }).round(2)
    
    print("\nBy Model:")
    print(summary)
    
    # Save summary
    summary_path = output_dir / "benchmark_summary.csv"
    summary.to_csv(summary_path)
    print(f"\n✓ Saved summary to: {summary_path}")
    
    # Create comparison plots
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        models = df['model'].unique()
        x_pos = np.arange(len(models))
        
        # Generation time
        time_data = df.groupby('model')['generation_time_s'].mean()
        axes[0, 0].bar(x_pos, time_data.values, color='blue', alpha=0.7)
        axes[0, 0].set_ylabel('Time (s)')
        axes[0, 0].set_title('Average Generation Time')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(models, rotation=15, ha='right')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # VRAM peak
        vram_data = df.groupby('model')['vram_peak_mb'].mean()
        axes[0, 1].bar(x_pos, vram_data.values, color='red', alpha=0.7)
        axes[0, 1].set_ylabel('VRAM (MB)')
        axes[0, 1].set_title('Average Peak VRAM')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(models, rotation=15, ha='right')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # GPU utilization
        gpu_data = df.groupby('model')['gpu_util_mean'].mean()
        axes[0, 2].bar(x_pos, gpu_data.values, color='green', alpha=0.7)
        axes[0, 2].set_ylabel('GPU Utilization (%)')
        axes[0, 2].set_title('Average GPU Usage')
        axes[0, 2].set_xticks(x_pos)
        axes[0, 2].set_xticklabels(models, rotation=15, ha='right')
        axes[0, 2].grid(axis='y', alpha=0.3)
        
        # CPU utilization
        cpu_data = df.groupby('model')['cpu_util_mean'].mean()
        axes[1, 0].bar(x_pos, cpu_data.values, color='orange', alpha=0.7)
        axes[1, 0].set_ylabel('CPU Utilization (%)')
        axes[1, 0].set_title('Average CPU Usage')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(models, rotation=15, ha='right')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Power
        power_data = df.groupby('model')['power_mean_watts'].mean()
        if power_data.sum() > 0:
            axes[1, 1].bar(x_pos, power_data.values, color='purple', alpha=0.7)
            axes[1, 1].set_ylabel('Power (W)')
            axes[1, 1].set_title('Average Power Draw')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(models, rotation=15, ha='right')
            axes[1, 1].grid(axis='y', alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Power data not available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        # Total energy
        energy_data = df.groupby('model')['energy_total_joules'].sum()
        if energy_data.sum() > 0:
            axes[1, 2].bar(x_pos, energy_data.values, color='brown', alpha=0.7)
            axes[1, 2].set_ylabel('Energy (J)')
            axes[1, 2].set_title('Total Energy Consumption')
            axes[1, 2].set_xticks(x_pos)
            axes[1, 2].set_xticklabels(models, rotation=15, ha='right')
            axes[1, 2].grid(axis='y', alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'Energy data not available', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
        
        plt.tight_layout()
        plot_path = output_dir / "benchmark_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved comparison plot to: {plot_path}")
        plt.close()
        
    except Exception as e:
        print(f"⚠ Could not create plots: {e}")
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - benchmark_results.csv (full data)")
    print(f"  - benchmark_summary.csv (aggregated)")
    print(f"  - benchmark_comparison.png (plots)")
    print(f"  - {len(PROMPTS)} prompt folders (each with seed folders containing model variants)")


if __name__ == "__main__":
    main()
