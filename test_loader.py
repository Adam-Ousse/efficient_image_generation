import torch
from pathlib import Path
from src.models import load_pipeline
import os

def test_standard_model():
    """Test loading standard FP16 model"""
    print("=" * 60)
    print("TEST: Standard FP16 Model")
    print("=" * 60)
    
    pipe = load_pipeline(
        model_id="black-forest-labs/FLUX.2-klein-4B",
        gguf_path=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16,
        enable_cpu_offload=False,
        enable_vae_slicing=True
    )
    
    # Generate test image
    image = pipe.generate(
        prompt="A cat holding a sign that says 'Hello World'",
        height=1024,
        width=1024,
        num_inference_steps=4,
        guidance_scale=3.5,
        seed=42,
        output_path="temp_test_outputs/test_output_standard.png"
    )
    
    print(f"[OK]Generated image size: {image.size}")
    print()


def test_gguf_model():
    """Test loading GGUF quantized model"""
    print("=" * 60)
    print("TEST: GGUF Quantized Model")
    print("=" * 60)
    
    # Use HuggingFace path format - will auto-download if needed
    model_id = "black-forest-labs/FLUX.2-klein-4B"
    # gguf_path = "unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q2_K.gguf"
    # gguf_path = "unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q4_K_M.gguf"
    gguf_path = "unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q8_0.gguf"
    pipe = load_pipeline(
        model_id=model_id,
        gguf_path=gguf_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16,
        enable_cpu_offload=False,
        enable_vae_slicing=True
    )
    
    # Generate test image
    image = pipe.generate(
        prompt="A penguin holding a sign that says 'GGUF Works!'",
        height=1024,
        width=1024,
        num_inference_steps=4,
        guidance_scale=3.5,
        seed=42,
        output_path=f"temp_test_outputs/test_output_gguf_{gguf_path.split('/')[-1].replace('.gguf', '')}.png"
    )
    
    print(f"[OK]Generated image size: {image.size}")
    print()


def test_multiple_generations():
    """Test generating multiple images with same pipeline"""
    print("=" * 60)
    print("TEST: Multiple Generations")
    print("=" * 60)
    
    pipe = load_pipeline(
        model_id="black-forest-labs/FLUX.2-klein-4B",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16
    )
    
    prompts = [
        "A red apple on a white table",
        "A blue butterfly on a flower",
        "A yellow taxi in New York"
    ]
    
    for i, prompt in enumerate(prompts):
        image = pipe.generate(
            prompt=prompt,
            height=512,  # Smaller for speed
            width=512,
            num_inference_steps=4,
            seed=i,
            output_path=f"temp_test_outputs/test_output_multi_{i}.png"
        )
        print(f"  [OK] Image {i+1}/{len(prompts)}: {image.size}")
    
    print()


if __name__ == "__main__":
    # Run tests
    os.makedirs("temp_test_outputs", exist_ok=True)
    # try:
    #     test_standard_model()
    # except Exception as e:
    #     print(f"ERROR Standard model test failed: {e}\n")
    
    try:
        test_gguf_model()
    except Exception as e:
        print(f"ERROR GGUF model test failed: {e}\n")
    
    # try:
    #     test_multiple_generations()
    # except Exception as e:
    #     print(f"ERROR Multiple generations test failed: {e}\n")
    
    print("=" * 60)
    print("Tests complete!")
    print("=" * 60)
