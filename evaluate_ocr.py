#!/usr/bin/env python3
"""
Evaluate OCR text accuracy in generated images
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from src.evaluation import evaluate_text_in_images, compare_models_ocr


# Configuration - Edit these parameters

OUTPUT_DIR = r'results/benchmark_latest'  # Directory containing benchmark outputs
REFERENCE_MODEL = 'FLUX2-Klein-dev-FP16'  # Reference model
# OCR_BACKEND = 'easyocr'  # OCR backend: 'easyocr' or 'glm' (GLM-OCR zai-org/GLM-OCR)
OCR_BACKEND = 'glm' 
LANGUAGES = ['en']  # OCR languages (easyocr only)
USE_GPU = True  # Use GPU for OCR
SAVE_CSV = 'ocr_results_distillation.csv'  # Save detailed results
SAVE_SUMMARY_CSV = 'ocr_summary_distillation.csv'  # Save aggregated results per model
SAVE_PLOT = 'ocr_plot_distillation.png'  # Save plot
models_to_evaluate = [
    'FLUX2-Klein-dev-FP16',
    "FLUX2-Klein-9B-FP16",
    "FLUX2-Klein-4B-FP16",
]
# Map prompt labels to expected text (if prompt names don't contain the text)
# Leave as None to auto-extract from prompt names with quotes
PROMPT_TEXT_MAPPING = None  

# Example custom mapping:
# PROMPT_TEXT_MAPPING = {
#     'A_sign_that_says_OPEN': 'OPEN',
#     'Storefront_with_COFFEE_SHOP': 'COFFEE SHOP',
# }

# PROMPT_TEXT_MAPPING={
#     "Classroom_with_Text" : "Laws of motion : Newton's Second Law: F = m x a",
    
    
# }

# Plotting

def plot_ocr_metrics(summary_df: pd.DataFrame, reference_model: str, save_path=None):
    """Plot OCR metrics ordered by quantization level"""
    
    plot_data = summary_df.copy()
    
    # Sort by quantization level (Q2 -> Q3 -> Q4 -> Q5 -> FP16)
    def quantization_order(model_name, quantization=None):
        # if quantization is none : Distillation order 
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
    
    # Extract short labels for x-axis
    def extract_quant_label(model_name):
        parts = model_name.split('-')
        if len(parts) > 0:
            return parts[-1]
        return model_name
    
    plot_data['label'] = plot_data['model'].apply(extract_quant_label)
    
    x = range(len(plot_data))
    labels = plot_data['label'].tolist()

    # Create 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    def _draw_metric(ax, means, stds, color, ylabel, title):
        means = means.to_numpy()
        stds  = stds.to_numpy()
        ax.fill_between(x,
                        means - stds, means + stds,
                        alpha=0.18, color=color, linewidth=0)
        ax.plot(x, means,
                marker='o', linewidth=2.5, markersize=8,
                color=color,
                markerfacecolor=color, markeredgecolor='white',
                markeredgewidth=1.5, label='mean ± std')
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_xlabel('Quantization Level', fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.legend(fontsize=10, loc='upper left')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

    # Plot 1: CER (lower is better)
    _draw_metric(axes[0],
                 plot_data['cer_mean'], plot_data['cer_std'],
                 '#e74c3c',
                 'Character Error Rate (CER)',
                 'CER vs Quantization\n(Lower is Better)')
    axes[0].set_ylim(0, 1.05)

    # Plot 2: CER Normalized (lower is better, always in [0,1])
    _draw_metric(axes[1],
                 plot_data['cer_normalized_mean'], plot_data['cer_normalized_std'],
                 '#9b59b6',
                 'CER Normalized',
                 'CER Normalized vs Quantization\n(Lower is Better, always ≤ 1)')
    axes[1].set_ylim(0, 1.05)

    # Plot 3: WER (lower is better)
    _draw_metric(axes[2],
                 plot_data['wer_mean'], plot_data['wer_std'],
                 '#f39c12',
                 'Word Error Rate (WER)',
                 'WER vs Quantization\n(Lower is Better)')
    axes[2].set_ylim(0, 1.05)
    
    # Add overall title
    fig.suptitle('OCR Metrics vs Quantization: FLUX.2-klein-4B', 
                 fontsize=16, fontweight='bold', y=1.02)
    
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
    print("OCR Text Evaluation")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"OCR backend: {OCR_BACKEND}")
    print(f"Languages: {LANGUAGES}")
    print(f"GPU: {USE_GPU}")
    print()
    
    # Evaluate all images
    print("Extracting text from images and computing metrics...")
    print("(This may take a while depending on number of images)")
    print()
    
    ocr_df = evaluate_text_in_images(
        output_dir=output_dir,
        prompt_text_mapping=PROMPT_TEXT_MAPPING,
        backend=OCR_BACKEND,
        languages=LANGUAGES,
        models_to_evaluate= models_to_evaluate,
        gpu=USE_GPU
    )
    
    if ocr_df.empty:
        print("\nNo images with expected text found for evaluation")
        return
    
    # Save detailed results
    if SAVE_CSV:
        csv_path = output_dir / SAVE_CSV
        ocr_df.to_csv(csv_path, index=False)
        print(f"\nDetailed results saved to {csv_path}")
    
    # Aggregate by model
    print("\n" + "="*80)
    print("OCR Metrics by Model (averaged across all prompts and seeds)")
    print("="*80)
    
    summary_df = compare_models_ocr(ocr_df)
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
    print("  CER (Character Error Rate): Lower is better (0 = perfect, can exceed 1)")
    print("  CER Normalized: Lower is better (always in [0,1])")
    print("  WER (Word Error Rate): Lower is better (0 = perfect)")
    print()
    print("  Best model for OCR: ", 
          summary_df.loc[summary_df['cer_mean'].idxmin(), 'model'])
    
    # Plot OCR metrics
    print("\n" + "="*80)
    print("Generating OCR plots...")
    print("="*80)
    save_plot_path = output_dir / SAVE_PLOT if SAVE_PLOT else None
    plot_ocr_metrics(summary_df, REFERENCE_MODEL, save_path=save_plot_path)


if __name__ == '__main__':
    main()
