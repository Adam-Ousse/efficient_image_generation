"""
Simple model loader for image generation pipelines
Supports both standard Diffusers and GGUF quantized models
"""

import torch
from pathlib import Path
from typing import Optional
from PIL import Image

try:
    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel, GGUFQuantizationConfig
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


class ImagePipeline:
    """Unified pipeline wrapper for image generation"""
    
    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.2-klein-4B",
        gguf_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        enable_cpu_offload: bool = False,
        enable_vae_slicing: bool = True
    ):
        """
        Initialize image generation pipeline
        
        Args:
            model_id: HuggingFace model ID
            gguf_path: Path to GGUF quantized file (if None, uses standard model)
            device: Device to run on ("cuda" or "cpu")
            dtype: Tensor dtype (torch.float16 or torch.bfloat16)
            enable_cpu_offload: Enable model CPU offloading to save VRAM
            enable_vae_slicing: Enable VAE slicing for lower memory usage
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers not installed. Run: pip install diffusers")
        
        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        self.gguf_path = gguf_path
        
        print(f"Loading model: {model_id}")
        if gguf_path:
            print(f"  GGUF quantization: {gguf_path}")
            # Resolve GGUF path (download if needed)
            self.gguf_path = self._resolve_gguf_path(gguf_path)
        print(f"  Device: {device}, Dtype: {dtype}")
        
        # Load transformer (GGUF or standard)
        transformer = self._load_transformer()
        
        # Load complete pipeline
        self.pipe = Flux2KleinPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            torch_dtype=dtype
        )
        
        # Apply optimizations
        if enable_cpu_offload:
            print("  Enabling CPU offload...")
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(device)
        
        if enable_vae_slicing and hasattr(self.pipe, 'enable_vae_slicing'):
            self.pipe.enable_vae_slicing()
            print("  VAE slicing enabled")
        else:
            print("  VAE slicing not available")
        
        print("✓ Pipeline ready")
    
    def _resolve_gguf_path(self, gguf_path: str) -> str:
        """
        Resolve GGUF path: download from HuggingFace if needed
        
        Supports formats:
        - Local path: "/path/to/model.gguf"
        - HF path: "repo_id/filename.gguf" (e.g., "unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q4_K_M.gguf")
        """
        # Check if it's a local path that exists
        if Path(gguf_path).exists():
            print(f"  ✓ Using local GGUF: {gguf_path}")
            return gguf_path
        
        # Check if it's a HuggingFace path (contains at least 2 slashes: repo/model/file.gguf)
        parts = gguf_path.split('/')
        if len(parts) >= 3:
            # Format: owner/repo/filename
            repo_id = f"{parts[0]}/{parts[1]}"
            filename = '/'.join(parts[2:])
            
            if not HF_HUB_AVAILABLE:
                raise ImportError(
                    "huggingface_hub not installed. Run: pip install huggingface_hub"
                )
            
            print(f"  Resolving GGUF from HuggingFace...")
            print(f"    Repo: {repo_id}")
            print(f"    File: {filename}")
            
            try:
                # First try to load from cache only (no download)
                try:
                    local_path = hf_hub_download(
                        repo_id=repo_id, 
                        filename=filename,
                        local_files_only=True
                    )
                    print(f"  ✓ Found in cache: {local_path}")
                    return local_path
                except Exception:
                    # Not in cache, need to download
                    print(f"  Downloading from HuggingFace...")
                    local_path = hf_hub_download(repo_id=repo_id, filename=filename)
                    print(f"  ✓ Downloaded to: {local_path}")
                    return local_path
                    
            except Exception as e:
                raise ValueError(
                    f"Failed to load GGUF from HuggingFace: {e}\n"
                    f"Repo: {repo_id}, File: {filename}"
                )
        
        # Path doesn't exist and isn't a HF path
        raise FileNotFoundError(
            f"GGUF file not found: {gguf_path}\n"
            f"Provide either:\n"
            f"  - A valid local path\n"
            f"  - A HuggingFace path (format: owner/repo/filename.gguf)"
        )
    
    def _load_transformer(self):
        """Load transformer with GGUF or standard weights"""
        if self.gguf_path and Path(self.gguf_path).exists():
            print("  Loading GGUF quantized transformer...")
            return Flux2Transformer2DModel.from_single_file(
                self.gguf_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype),
                torch_dtype=self.dtype,
                config=self.model_id,
                subfolder="transformer"
            )
        else:
            print("  Loading standard transformer...")
            return Flux2Transformer2DModel.from_pretrained(
                self.model_id,
                subfolder="transformer",
                torch_dtype=self.dtype
            )
    
    def generate(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 4,
        guidance_scale: float = 3.5,
        seed: Optional[int] = None,
        output_path: Optional[str] = None
    ) -> Image.Image:
        """
        Generate image from text prompt
        
        Args:
            prompt: Text description
            height: Image height
            width: Image width
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance strength
            seed: Random seed for reproducibility
            output_path: Save path (if provided)
        
        Returns:
            PIL Image
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        print(f"\nGenerating: '{prompt}'")
        print(f"  Resolution: {width}x{height}, Steps: {num_inference_steps}")
        
        image = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images[0]
        
        if output_path:
            image.save(output_path)
            print(f"  Saved to: {output_path}")
        
        return image


def load_pipeline(
    model_id: str = "black-forest-labs/FLUX.2-klein-4B",
    gguf_path: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    enable_cpu_offload: bool = False,
    enable_vae_slicing: bool = True
) -> ImagePipeline:
    """
    Load an image generation pipeline
    
    Args:
        model_id: HuggingFace model ID
        gguf_path: Path to GGUF file for quantized model (optional)
        device: Device ("cuda" or "cpu")
        dtype: Tensor dtype (torch.float16 or torch.bfloat16)
        enable_cpu_offload: Enable CPU offloading
        enable_vae_slicing: Enable VAE slicing
    
    Returns:
        ImagePipeline instance with .generate() method
    
    Example:
        >>> pipe = load_pipeline(model_id="black-forest-labs/FLUX.2-klein-4B")
        >>> image = pipe.generate("A cat in a hat", seed=42)
        
        >>> # With GGUF quantization
        >>> pipe = load_pipeline(
        ...     model_id="black-forest-labs/FLUX.2-klein-4B",
        ...     gguf_path="/path/to/model.gguf"
        ... )
    """
    return ImagePipeline(
        model_id=model_id,
        gguf_path=gguf_path,
        device=device,
        dtype=dtype,
        enable_cpu_offload=enable_cpu_offload,
        enable_vae_slicing=enable_vae_slicing
    )
