from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from src.evaluation import compare_models_fid


# ============================================================================
# Configuration - Edit these parameters
# ============================================================================

OUTPUT_DIR = 'benchmark_results_20260211_022815'  # Directory containing benchmark outputs
REFERENCE_MODEL = 'FLUX2-Klein-FP16'  # Reference model to compare against
DEVICE = 'cuda'  # Device to use for computation
BATCH_SIZE = 32  # Batch size for feature extraction
SAVE_CSV = 'fid_results.csv'  # CSV file to save results (None to skip)
SAVE_PLOT = 'fid_plot.png'  # PNG file to save plot (None to skip)


# ============================================================================
# Plotting
# ============================================================================

def plot_fid_scores(results_df: pd.DataFrame, reference_model: str, save_path=None):
    """Plot FID scores ordered by quantization level"""
    
    # Add reference model with FID=0 (comparing to itself)
    plot_data = results_df.copy()
    ref_row = pd.DataFrame([{
        'model': reference_model,
        'fid_score': 0.0
    }])
    plot_data = pd.concat([plot_data, ref_row], ignore_index=True)
    
    # Sort by quantization level (Q2 -> Q4 -> FP16)
    # Extract quantization info for sorting
    def quantization_order(model_name):
        if 'Q2' in model_name:
            return 0
        elif 'Q4' in model_name:
            return 1
        else:  # FP16 or other
            return 2
    
    plot_data['sort_key'] = plot_data['model'].apply(quantization_order)
    plot_data = plot_data.sort_values('sort_key')
    
    # Extract short labels for x-axis
    def extract_quant_label(model_name):
        if 'Q2_K' in model_name:
            return 'Q2_K'
        elif 'Q4_K_M' in model_name:
            return 'Q4_K_M'
        elif 'FP16' in model_name:
            return 'FP16'
        else:
            return model_name
    
    plot_data['label'] = plot_data['model'].apply(extract_quant_label)
    
    # Create plot with similar style to the MMLU plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot line with markers
    ax.plot(plot_data['label'], plot_data['fid_score'], 
            marker='o', linewidth=2.5, markersize=8, 
            color='#27ae60', label='GGUF Quantized',
            markerfacecolor='#27ae60', markeredgecolor='white', 
            markeredgewidth=1.5)
    
    # Add horizontal line for reference (FP16 baseline)
    fp16_score = plot_data[plot_data['label'] == 'FP16']['fid_score'].values[0]
    ax.axhline(y=fp16_score, color='#3498db', linestyle='--', 
               linewidth=2, label=f'{reference_model}', alpha=0.8)
    
    # Styling
    ax.set_xlabel('Quantization Level', fontsize=13)
    ax.set_ylabel('FID Score (distortion)', fontsize=13)
    ax.set_title('FID Score vs Quantization: FLUX.2-klein-4B', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Rotate x-labels
    plt.xticks(rotation=0, ha='center')
    
    # Add legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Add note similar to the MMLU plot
    note_text = "Note: Plots closer to the blue (FP16) line means better quality.\nLower FID scores indicate better image quality."
    ax.text(0.02, 0.98, note_text, 
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Adjust y-axis to show more detail
    y_min = min(plot_data['fid_score']) - (max(plot_data['fid_score']) * 0.1)
    y_max = max(plot_data['fid_score']) * 1.15
    ax.set_ylim(max(0, y_min), y_max)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


# ============================================================================
# Main evaluation
# ============================================================================

def main():
    output_dir = Path(OUTPUT_DIR)
    
    if not output_dir.exists():
        print(f"Error: Output directory {output_dir} does not exist")
        print("Run benchmark_models.py first to generate images")
        return
    
    print("="*80)
    print("FID Evaluation")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Reference model: {REFERENCE_MODEL}")
    print(f"Device: {DEVICE}")
    print()
    
    # Compute FID scores
    results_df = compare_models_fid(
        output_dir=output_dir,
        reference_model=REFERENCE_MODEL,
        device=DEVICE,
        batch_size=BATCH_SIZE
    )
    
    # Display results
    print("\n" + "="*80)
    print("FID Results (lower is better)")
    print("="*80)
    print(results_df.to_string(index=False))
    print()
    
    # Save to CSV if requested
    if SAVE_CSV:
        csv_path = Path(SAVE_CSV)
        results_df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
    
    # Summary statistics
    print("\nSummary:")
    print(f"  Best model: {results_df.loc[results_df['fid_score'].idxmin(), 'model']}")
    print(f"  Best FID: {results_df['fid_score'].min():.2f}")
    print(f"  Worst model: {results_df.loc[results_df['fid_score'].idxmax(), 'model']}")
    print(f"  Worst FID: {results_df['fid_score'].max():.2f}")
    print(f"  Average FID: {results_df['fid_score'].mean():.2f}")
    
    # Plot FID scores
    print("\n" + "="*80)
    print("Generating FID plot...")
    print("="*80)
    plot_fid_scores(results_df, REFERENCE_MODEL, save_path=SAVE_PLOT)


if __name__ == '__main__':
    main()
