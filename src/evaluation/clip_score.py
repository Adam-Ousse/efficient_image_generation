"""CLIP score metric implementation for text-image alignment"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path
import pandas as pd
from typing import Union, List

class ClipEvaluator:
    """Evaluates text-to-image alignment using OpenAI's CLIP"""
    
    def __init__(self, model_id="openai/clip-vit-base-patch32", device='cuda'):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model.eval()
        
    @torch.no_grad()
    def compute_scores(self, images: List[Image.Image], texts: List[str]) -> List[float]:
        """Compute CLIP scores for a batch of image-text pairs"""
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        # We want the diagonal which represents the score for the paired image and text
        logits_per_image = outputs.logits_per_image
        scores = logits_per_image.diag().cpu().tolist()
        
        return scores

def compare_models_clip(output_dir: Union[str, Path],
                        device: str = 'cuda',
                        batch_size: int = 32,
                        models_to_evaluate: List[str] = None) -> pd.DataFrame:
    """
    Compute CLIP scores for all models in the output directory.
    Assumes directory structure: output_dir / prompt / seed_X / model_name.png
    """
    output_dir = Path(output_dir)
    prompt_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    
    if not prompt_dirs:
        raise ValueError(f"No prompt directories found in {output_dir}")
        
    evaluator = ClipEvaluator(device=device)
    results = []
    
    print("Computing CLIP scores for text-image alignment...")
    
    # We will gather all images and prompts, then process them in batches
    for prompt_dir in prompt_dirs:
        # Assuming the folder name is the prompt (replacing underscores if slugified)
        prompt_text = prompt_dir.name.replace('_', ' ') 
        
        for seed_dir in prompt_dir.glob('seed_*'):
            if not seed_dir.is_dir():
                continue
                
            for img_path in seed_dir.glob('*.png'):
                model_name = img_path.stem
                if models_to_evaluate and model_name not in models_to_evaluate:
                    continue
                try:
                    img = Image.open(img_path).convert('RGB')
                    # Compute score for a single pair (can be optimized to batch process if memory allows)
                    score = evaluator.compute_scores([img], [prompt_text])[0]
                    
                    results.append({
                        'model': model_name,
                        'prompt': prompt_text,
                        'seed': seed_dir.name,
                        'clip_score': score
                    })
                except Exception as e:
                    print(f"Warning: Failed to process {img_path}: {e}")

    df = pd.DataFrame(results)
    
    # Aggregate by model
    if not df.empty:
        summary_df = df.groupby('model').agg(
            clip_mean=('clip_score', 'mean'),
            clip_std=('clip_score', 'std'),
            num_images=('clip_score', 'count')
        ).reset_index()
        return df, summary_df
    else:
        return pd.DataFrame(), pd.DataFrame()