"""SSIM (Structural Similarity Index) metric implementation"""

from pathlib import Path
from typing import Union, List, Dict
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def load_image_as_array(image_path: Union[str, Path]) -> np.ndarray:
    """Load image and convert to numpy array"""
    img = Image.open(image_path).convert('RGB')
    return np.array(img)


def compute_ssim(image1_path: Union[str, Path], 
                 image2_path: Union[str, Path],
                 multichannel: bool = True) -> float:
    """
    Compute SSIM between two images
    
    Args:
        image1_path: Path to first image
        image2_path: Path to second image
        multichannel: Whether to compute SSIM for color images (True) or grayscale (False)
    
    Returns:
        SSIM score (higher is better, 1.0 = identical)
    """
    img1 = load_image_as_array(image1_path)
    img2 = load_image_as_array(image2_path)
    
    # Ensure images have same shape
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes don't match: {img1.shape} vs {img2.shape}")
    
    # Compute SSIM
    score = ssim(img1, img2, channel_axis=2 if multichannel else None, data_range=255)
    
    return float(score)


def evaluate_ssim_in_images(output_dir: Union[str, Path],
                            reference_model: str = "FLUX2-Klein-FP16",
                            models_to_evaluate: List[str] = None) -> pd.DataFrame:
    """
    Evaluate SSIM for all generated images against reference model
    
    Args:
        output_dir: Directory with prompt/seed/model structure
        reference_model: Name of reference model to compare against
        models_to_evaluate: List of models to evaluate (if None, evaluate all models)
    
    Returns:
        DataFrame with SSIM scores for each image
    """
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        raise ValueError(f"Output directory not found: {output_dir}")
    
    results = []
    
    # Iterate through prompt folders
    prompt_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])
    
    for prompt_dir in prompt_dirs:
        prompt_label = prompt_dir.name
        
        print(f"\nEvaluating prompt: {prompt_label}")
        
        # Iterate through seed folders
        seed_dirs = sorted([d for d in prompt_dir.iterdir() if d.is_dir()])
        
        for seed_dir in seed_dirs:
            seed = seed_dir.name
            
            # Find reference image
            ref_image = seed_dir / f"{reference_model}.png"
            if not ref_image.exists():
                print(f"  Warning: Reference image not found: {ref_image}")
                continue
            
            # Iterate through model images
            image_files = sorted(seed_dir.glob('*.png'))
            
            for image_path in image_files:
                model_name = image_path.stem
                
                # Skip comparing reference to itself
                if model_name == reference_model:
                    continue
                
                # Skip models not in the evaluation list
                if models_to_evaluate and model_name not in models_to_evaluate:
                    continue
                
                try:
                    ssim_score = compute_ssim(ref_image, image_path)
                    
                    results.append({
                        'prompt': prompt_label,
                        'seed': seed,
                        'model': model_name,
                        'reference': reference_model,
                        'ssim': ssim_score
                    })
                    
                    print(f"  {model_name} ({seed}): SSIM={ssim_score:.4f}")
                
                except Exception as e:
                    print(f"  Error evaluating {image_path}: {e}")
                    continue
    
    df = pd.DataFrame(results)
    return df


def compare_models_ssim(ssim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate SSIM metrics by model across all prompts and seeds
    
    Args:
        ssim_df: DataFrame from evaluate_ssim_in_images()
    
    Returns:
        DataFrame with average SSIM per model
    """
    summary = ssim_df.groupby('model').agg({
        'ssim': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    # Sort by SSIM mean (higher is better)
    summary = summary.sort_values('ssim_mean', ascending=False)
    
    return summary
