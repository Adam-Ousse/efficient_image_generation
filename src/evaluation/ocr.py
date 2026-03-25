"""OCR-based text evaluation for image generation"""

import re
from pathlib import Path
from typing import List, Dict, Union, Tuple
import pandas as pd
from PIL import Image
import Levenshtein


# OCR Backends

class EasyOCRBackend:
    """EasyOCR backend for text extraction"""
    
    def __init__(self, languages=['en'], gpu=True):
        import easyocr
        print(f"Initializing EasyOCR reader for languages: {languages}")
        self.reader = easyocr.Reader(languages, gpu=gpu)
    
    def extract_text(self, image_path: Union[str, Path]) -> str:
        results = self.reader.readtext(str(image_path))
        return ' '.join(text for (bbox, text, conf) in results)


class GLMOCRBackend:
    """GLM-OCR backend (zai-org/GLM-OCR) for text extraction"""
    
    MODEL_PATH = "zai-org/GLM-OCR"
    
    def __init__(self, device_map='auto'):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        import torch
        
        print(f"Initializing GLM-OCR model from {self.MODEL_PATH}...")
        self.processor = AutoProcessor.from_pretrained(self.MODEL_PATH)
        self.model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path=self.MODEL_PATH,
            torch_dtype='auto',
            device_map=device_map,
        )
        print("GLM-OCR model loaded.")
    
    def extract_text(self, image_path: Union[str, Path]) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": str(image_path)},
                    {"type": "text", "text": "Recognize the main text in the image:"}
                ],
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        inputs.pop("token_type_ids", None)
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=8192)
        output_text = self.processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return output_text.strip()


# OCR Evaluator

class OCREvaluator:
    """Evaluate text presence and accuracy in generated images"""
    
    def __init__(self, backend='easyocr', languages=['en'], gpu=True, device_map='auto'):
        """
        Initialize OCR evaluator
        
        Args:
            backend: 'easyocr' or 'glm' (GLM-OCR)
            languages: Language codes for EasyOCR (ignored for GLM-OCR)
            gpu: Use GPU for EasyOCR
            device_map: Device map for GLM-OCR
        """
        self.backend_name = backend
        
        if backend == 'glm':
            self.backend = GLMOCRBackend(device_map=device_map)
        else:
            self.backend = EasyOCRBackend(languages=languages, gpu=gpu)
    
    def extract_text(self, image_path: Union[str, Path]) -> str:
        """Extract all text from image using configured OCR backend"""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self.backend.extract_text(image_path)
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove unicode symbols, scribbles, checkmarks, math symbols, arrows, etc.
        # Keep only ASCII letters, digits, and spaces
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII (×, ✓, ', etc.)
        text = re.sub(r'[^\w\s]', '', text)           # Remove remaining punctuation
        text = ' '.join(text.split())                  # Collapse whitespace again
        return text
    
    def calculate_cer(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Character Error Rate (CER)
        
        CER = (insertions + deletions + substitutions) / len(reference)
        """
        ref_norm = self.normalize_text(reference)
        hyp_norm = self.normalize_text(hypothesis)
        
        if len(ref_norm) == 0:
            return 0.0 if len(hyp_norm) == 0 else 1.0
        
        distance = Levenshtein.distance(ref_norm, hyp_norm)
        cer = distance / len(ref_norm)
        
        return cer
    def calculate_cer_normalized(self, reference: str, hypothesis: str) -> float:
        """
        Normalized CER = (S + D + I) / (S + D + I + C)
        where C = number of correct characters.
        Always in [0, 1]. More interpretable than standard CER.
        
        0.0 = perfect, 1.0 = all operations were errors (nothing correct)
        """
        ref_norm = self.normalize_text(reference)
        hyp_norm = self.normalize_text(hypothesis)

        if len(ref_norm) == 0 and len(hyp_norm) == 0:
            return 0.0
        if len(ref_norm) == 0:
            return 1.0

        edits = Levenshtein.distance(ref_norm, hyp_norm)  # S + D + I
        correct = len(ref_norm) - edits                    # C (approximate)
        correct = max(0, correct)                          # clamp to 0

        denominator = edits + correct                      # S + D + I + C
        if denominator == 0:
            return 0.0

        return edits / denominator
    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate (WER)
        
        WER = (insertions + deletions + substitutions) / num_words_in_reference
        """
        ref_norm = self.normalize_text(reference)
        hyp_norm = self.normalize_text(hypothesis)
        
        ref_words = ref_norm.split()
        hyp_words = hyp_norm.split()
        
        if len(ref_words) == 0:
            return 0.0 if len(hyp_words) == 0 else 1.0
        
        # Calculate word-level Levenshtein distance
        distance = Levenshtein.distance(ref_words, hyp_words)
        wer = distance / len(ref_words)
        
        return wer
    
    
    def evaluate_image(self, image_path: Union[str, Path], 
                      expected_text: str) -> Dict[str, float]:
        """
        Evaluate a single image against expected text
        
        Args:
            image_path: Path to generated image
            expected_text: Text that should appear in the image
        
        Returns:
            Dictionary with metrics: cer, wer, text_presence, extracted_text
        """
        extracted_text = self.extract_text(image_path)
        
        cer = self.calculate_cer(expected_text, extracted_text)
        cer_norm = self.calculate_cer_normalized(expected_text, extracted_text)
        wer = self.calculate_wer(expected_text, extracted_text)
        
        return {
            'extracted_text': extracted_text,
            'cer': cer,
            'cer_normalized': cer_norm,
            'wer': wer
        }


def read_prompt_from_file(prompt_dir: Path) -> str:
    """
    Read prompt text from prompt.txt file in prompt directory
    
    Args:
        prompt_dir: Path to prompt directory
    
    Returns:
        Prompt text or empty string if file not found
    """
    prompt_file = prompt_dir / "prompt.txt"
    if prompt_file.exists():
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return ""


def extract_expected_text_from_prompt(prompt: str) -> str:
    """
    Extract expected text from prompt by finding quoted strings
    
    Example: 'A sign that says "OPEN"' -> 'OPEN'
    """
    # Find text in double quotes
    matches = re.findall(r'"([^"]*)"', prompt)
    if matches:
        return ' '.join(matches)
    
    # Find text in single quotes
    matches = re.findall(r"'([^']*)'", prompt)
    if matches:
        return ' '.join(matches)
    
    # # Find text after "says" or "reads"
    # says_match = re.search(r'(?:says|reads)\s+([A-Z][A-Z\s]+)', prompt)
    # if says_match:
    #     return says_match.group(1).strip()
    print("Failed to extract expected text from prompt.")
    return ""


def evaluate_text_in_images(output_dir: Union[str, Path],
                            prompt_text_mapping: Dict[str, str] = None,
                            backend: str = 'easyocr',
                            languages=['en'],
                            gpu=True,
                            models_to_evaluate: List[str] = None,
                            device_map='auto') -> pd.DataFrame:
    """
    Evaluate OCR text in all generated images
    
    Args:
        output_dir: Directory with prompt/seed/model structure
        prompt_text_mapping: Dict mapping prompt labels to expected text
                            If None, will try to extract from prompt names
        backend: OCR backend to use - 'easyocr' or 'glm'
        languages: Languages for EasyOCR (ignored for GLM-OCR)
        gpu: Use GPU for EasyOCR
        models_to_evaluate: List of specific models to evaluate (if None, evaluate all)
        device_map: Device map for GLM-OCR
    
    Returns:
        DataFrame with OCR metrics for each image
    """
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        raise ValueError(f"Output directory not found: {output_dir}")
    
    # Initialize evaluator
    evaluator = OCREvaluator(backend=backend, languages=languages, gpu=gpu, device_map=device_map)
    
    results = []
    
    # Iterate through prompt folders
    prompt_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])
    
    for prompt_dir in prompt_dirs:
        prompt_label = prompt_dir.name
        
        # Get expected text
        if prompt_text_mapping and prompt_label in prompt_text_mapping:
            expected_text = prompt_text_mapping[prompt_label]
        else:
            # First try to read from prompt.txt file
            prompt_text = read_prompt_from_file(prompt_dir)
            if prompt_text:
                # Extract text from the actual prompt
                expected_text = extract_expected_text_from_prompt(prompt_text)
            else:
                # Fallback: try to extract from folder name
                raise ValueError(f"No expected text found for prompt '{prompt_label}' and no prompt.txt file")
        
        if not expected_text:
            print(f"Warning: No expected text for prompt '{prompt_label}', skipping OCR eval")
            continue
        
        print(f"\nEvaluating prompt: {prompt_label}")
        print(f"  Expected text: '{expected_text}'")
        
        # Iterate through seed folders
        seed_dirs = sorted([d for d in prompt_dir.iterdir() if d.is_dir()])
        
        for seed_dir in seed_dirs:
            seed = seed_dir.name
            
            # Iterate through model images
            image_files = sorted(seed_dir.glob('*.png'))
            
            for image_path in image_files:
                model_name = image_path.stem
                
                if models_to_evaluate is not None and model_name not in models_to_evaluate:
                    print(f"  Skipping {model_name} (not in models_to_evaluate)")
                    continue
                
                try:
                    metrics = evaluator.evaluate_image(image_path, expected_text)
                    
                    results.append({
                        'prompt': prompt_label,
                        'seed': seed,
                        'model': model_name,
                        'expected_text': expected_text,
                        'extracted_text': metrics['extracted_text'],
                        'cer': metrics['cer'],
                        'cer_normalized': metrics['cer_normalized'],
                        'wer': metrics['wer'],
                    })
                    
                    print(f"    {model_name} ({seed}): CER={metrics['cer']:.3f}, "
                          f"CER_norm={metrics['cer_normalized']:.3f}, WER={metrics['wer']:.3f}")
                
                except Exception as e:
                    print(f"    Error evaluating {image_path}: {e}")
                    continue
    
    df = pd.DataFrame(results)
    return df


def compare_models_ocr(ocr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate OCR metrics by model across all prompts and seeds
    
    Args:
        ocr_df: DataFrame from evaluate_text_in_images()
    
    Returns:
        DataFrame with average metrics per model
    """
    summary = ocr_df.groupby('model').agg({
        'cer': ['mean', 'std'],
        'cer_normalized': ['mean', 'std'],
        'wer': ['mean', 'std'],
    }).round(4)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    # Sort by CER (lower is better)
    summary = summary.sort_values('cer_mean')
    
    return summary
