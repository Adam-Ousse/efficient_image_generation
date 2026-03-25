#!/usr/bin/env python3
"""
Evaluate image generation models using SSIM
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from src.evaluation import evaluate_ssim_in_images, compare_models_ssim


# Configuration - Edit these parameters

OUTPUT_DIR = r'results/benchmark_flux_qwen_latest'  # Directory containing benchmark outputs
REFERENCE_MODEL = 'FLUX-4B-FP16__Qwen-4B-BF16'  # Reference model to compare against
SAVE_CSV = 'ssim_results.csv'  # Save detailed results
SAVE_SUMMARY_CSV = 'ssim_summary.csv'  # Save aggregated results per model
SAVE_PLOT = 'ssim_plot.png'  # Save plot


# Plotting

def plot_ssim_scores(summary_df: pd.DataFrame, reference_model: str, save_path=None):
    """Plot SSIM scores ordered by quantization level"""
    
    plot_data = summary_df.copy()
    
    # Sort by quantization level (Q2 -> Q3 -> Q4 -> Q5 -> FP16)
    def quantization_order(model_name):
        if 'FP16' in model_name or 'FP32' in model_name:
            return 100
        import re
        quant_match = re.search(r'Q(\d+)', model_name)
        if quant_match:
            return int(quant_match.group(1))
        return 50
    
    plot_data['sort_key'] = plot_data['model'].apply(quantization_order)
    plot_data = plot_data.sort_values('sort_key')
    
    # Extract short labels for x-axis
    def extract_quant_label(model_name):
        parts = model_name.split('-')
        if len(parts) > 0:
            return parts[-1]
        return model_name
    
    plot_data['label'] = plot_data['model'].apply(extract_quant_label)

    x      = range(len(plot_data))
    labels = plot_data['label'].tolist()
    means  = plot_data['ssim_mean'].to_numpy()
    stds   = plot_data['ssim_std'].to_numpy()
    color  = '#9b59b6'

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Shaded std band
    ax.fill_between(x, means - stds, means + stds,
                    alpha=0.18, color=color, linewidth=0)

    # Plot line with markers
    ax.plot(x, means,
            marker='o', linewidth=2.5, markersize=8,
            color=color, label='mean ± std',
            markerfacecolor=color, markeredgecolor='white',
            markeredgewidth=1.5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)

    # Styling
    ax.set_xlabel('Quantization Level', fontsize=13)
    ax.set_ylabel('SSIM Score (structural similarity)', fontsize=13)
    ax.set_title('SSIM vs Quantization: FLUX.2-klein-4B',
                 fontsize=16, fontweight='bold', pad=20)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(fontsize=11, loc='lower right')

    # Rotate x-labels
    plt.xticks(rotation=0, ha='center')

    # Set y-axis limits (SSIM is between 0 and 1)
    ax.set_ylim(0, 1.05)
    
    # Add note
    note_text = f"Note: Higher SSIM means more similar to {reference_model}.\nSSIM = 1.0 means identical images."
    ax.text(0.02, 0.02, note_text, 
            transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


# Main evaluation

def main():
    output_dir = Path(OUTPUT_DIR)
    
    if not output_dir.exists():
        print(f"Error: Output directory {output_dir} does not exist")
        print("Run benchmark_models.py first to generate images")
        return
    
    print("="*80)
    print("SSIM Evaluation")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Reference model: {REFERENCE_MODEL}")
    print()
    
    # Compute SSIM scores
    print("Computing SSIM between images...")
    ssim_df = evaluate_ssim_in_images(
        output_dir=output_dir,
        reference_model=REFERENCE_MODEL
    )
    
    if ssim_df.empty:
        print("\nNo images found for evaluation")
        return
    
    # Save detailed results
    if SAVE_CSV:
        csv_path = output_dir / SAVE_CSV
        ssim_df.to_csv(csv_path, index=False)
        print(f"\nDetailed results saved to {csv_path}")
    
    # Aggregate by model
    print("\n" + "="*80)
    print("SSIM Metrics by Model (averaged across all prompts and seeds)")
    print("="*80)
    
    summary_df = compare_models_ssim(ssim_df)
    print(summary_df.to_string(index=False))
    
    # Save summary
    if SAVE_SUMMARY_CSV:
        summary_path = output_dir / SAVE_SUMMARY_CSV
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")
    
    # Interpretation
    print("\n" + "="*80)
    print("Metric Interpretation:")
    print("="*80)
    print("  SSIM (Structural Similarity): Higher is better (1.0 = identical)")
    print(f"  Measures perceptual similarity to {REFERENCE_MODEL}")
    print()
    print("  Best model: ", 
          summary_df.loc[summary_df['ssim_mean'].idxmax(), 'model'])
    print(f"  Best SSIM: {summary_df['ssim_mean'].max():.4f}")
    
    # Plot SSIM scores
    print("\n" + "="*80)
    print("Generating SSIM plot...")
    print("="*80)
    save_plot_path = output_dir / SAVE_PLOT if SAVE_PLOT else None
    plot_ssim_scores(summary_df, REFERENCE_MODEL, save_path=save_plot_path)


if __name__ == '__main__':
    main()
