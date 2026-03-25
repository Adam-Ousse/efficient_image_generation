#!/usr/bin/env python3
"""Evaluate models with SSIM, FID, and CLIP metrics."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from src.evaluation import evaluate_ssim_in_images, compare_models_ssim
from src.evaluation.fid import compare_models_fid
from src.evaluation.clip_score import compare_models_clip

OUTPUT_DIR = r'results/benchmark_latest'  # Directory containing benchmark outputs
REFERENCE_MODEL = 'FLUX2-Klein-dev-FP16'  # Reference model to compare against (SSIM/FID)
SAVE_SUMMARY_CSV = 'comprehensive_metrics_summary_distill.csv'
models_to_evaluate = [
    'FLUX2-Klein-dev-FP16',
    "FLUX2-Klein-9B-FP16",
    "FLUX2-Klein-4B-FP16",
]
DEVICE = 'cuda'
print("Configuration:")
print(f"  Output Directory: {OUTPUT_DIR}")
print(f"  Reference Model: {REFERENCE_MODEL}")
print(f"  Device: {DEVICE}")
# Plotting

def plot_metric_scores(summary_df: pd.DataFrame, metric_name: str, y_col_mean: str, y_col_std: str = None, reference_model: str = None, save_path=None):
    """Plot metric scores ordered by quantization level"""
    if summary_df.empty or y_col_mean not in summary_df.columns:
        return

    plot_data = summary_df.copy()
    
    def quantization_order(model_name, quantization=None):
        if quantization:
            if 'FP16' in model_name or 'FP32' in model_name:
                return 100
            import re
            quant_match = re.search(r'Q(\d+)', model_name)
            if quant_match:
                return int(quant_match.group(1))
            return 50
        else: 
            if "dev" in model_name:
                return 100
            elif "9b" in model_name:
                return 90
            elif "4b" in model_name:
                return 80
    
    plot_data['sort_key'] = plot_data['model'].apply(quantization_order, quantization=False)
    plot_data = plot_data.sort_values('sort_key')
    
    def extract_quant_label(model_name):
        parts = model_name.split('-')
        return parts[-1] if len(parts) > 0 else model_name
    
    plot_data['label'] = plot_data['model'].apply(extract_quant_label)

    x      = range(len(plot_data))
    labels = plot_data['label'].tolist()
    means  = plot_data[y_col_mean].to_numpy()
    
    color_map = {'SSIM': '#9b59b6', 'FID': '#e74c3c', 'CLIP': '#2ecc71'}
    color = color_map.get(metric_name, '#34495e')

    fig, ax = plt.subplots(figsize=(12, 7))

    if y_col_std and y_col_std in plot_data.columns:
        stds = plot_data[y_col_std].fillna(0).to_numpy()
        ax.fill_between(x, means - stds, means + stds, alpha=0.18, color=color, linewidth=0)
        label_text = 'mean ± std'
    else:
        label_text = 'score'

    ax.plot(x, means, marker='o', linewidth=2.5, markersize=8, color=color, 
            label=label_text, markerfacecolor=color, markeredgecolor='white', markeredgewidth=1.5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Quantization Level', fontsize=13)
    ax.set_ylabel(f'{metric_name} Score', fontsize=13)
    ax.set_title(f'{metric_name} vs Quantization', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11, loc='best')
    plt.xticks(rotation=0, ha='center')

    if metric_name == 'SSIM':
        ax.set_ylim(0, 1.05)
        note = f"Higher is better (more structurally similar to {reference_model})."
    elif metric_name == 'FID':
        note = f"Lower is better (closer distribution to {reference_model})."
    else: # CLIP
        note = "Higher is better (better text-to-image alignment)."

    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()

def main():
    output_dir = Path(OUTPUT_DIR)
    
    if not output_dir.exists():
        print(f"Error: Output directory {output_dir} does not exist.")
        return
    
    print("="*80)
    print("Comprehensive Image Generation Evaluation (SSIM, FID, CLIP)")
    print("="*80)
    
    final_summary_df = pd.DataFrame()

    print("\n[1/3] Computing SSIM...")
    ssim_df = evaluate_ssim_in_images(output_dir=output_dir, reference_model=REFERENCE_MODEL, models_to_evaluate=models_to_evaluate)
    if not ssim_df.empty:
        ssim_summary = compare_models_ssim(ssim_df)
        final_summary_df = ssim_summary[['model', 'ssim_mean', 'ssim_std']]
        plot_metric_scores(ssim_summary, 'SSIM', 'ssim_mean', 'ssim_std', REFERENCE_MODEL, output_dir / 'ssim_plot.png')

    print("\n[2/3] Computing FID...")
    try:
        fid_df = compare_models_fid(output_dir=output_dir, reference_model=REFERENCE_MODEL, device=DEVICE, models_to_evaluate=models_to_evaluate)
        if not fid_df.empty:
            if final_summary_df.empty:
                final_summary_df = fid_df[['model', 'fid_score']]
            else:
                final_summary_df = pd.merge(final_summary_df, fid_df[['model', 'fid_score']], on='model', how='outer')
            plot_metric_scores(fid_df, 'FID', 'fid_score', reference_model=REFERENCE_MODEL, save_path=output_dir / 'fid_plot.png')
    except Exception as e:
        print(f"FID Evaluation failed or was skipped: {e}")

    print("\n[3/3] Computing CLIP scores...")
    try:
        clip_raw_df, clip_summary_df = compare_models_clip(output_dir=output_dir, device=DEVICE, models_to_evaluate=models_to_evaluate)
        if not clip_summary_df.empty:
            if final_summary_df.empty:
                final_summary_df = clip_summary_df[['model', 'clip_mean', 'clip_std']]
            else:
                final_summary_df = pd.merge(final_summary_df, clip_summary_df[['model', 'clip_mean', 'clip_std']], on='model', how='outer')
            plot_metric_scores(clip_summary_df, 'CLIP', 'clip_mean', 'clip_std', save_path=output_dir / 'clip_plot.png')
    except Exception as e:
        print(f"CLIP Evaluation failed or was skipped: {e}")

    if not final_summary_df.empty:
        print("\n" + "="*80)
        print("Final Consolidated Metrics by Model")
        print("="*80)
        print(final_summary_df.to_string(index=False))
        
        summary_path = output_dir / SAVE_SUMMARY_CSV
        final_summary_df.to_csv(summary_path, index=False)
        print(f"\nConsolidated summary saved to {summary_path}")
    else:
        print("\nNo metrics were successfully computed.")

if __name__ == '__main__':
    main()