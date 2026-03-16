"""
Abstract base for model descriptors.
Override memory_usage_factor, and the three abstract methods.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class ModelBase:
    """
    Abstract descriptor for a diffusion model.

    Knows how to:
      1. Load the transformer (HuggingFace hub or local GGUF checkpoint).
      2. Estimate how much VRAM to reserve for activations at a given resolution
         (mirrors ComfyUI's model_base.memory_required()).

    Attributes
    ----------
    _empirical_extra_gb_1024 : float
        vram_headroom
    vae_latent_factor : int
        Spatial downscale applied by the VAE encoder (8 for FLUX/SD1.5/SDXL).
    """
    _empirical_extra_gb_1024: float = 0.8  # emperical vram usage headroom
    def __init__(
        self,
        model_id: str,
        gguf_path: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Parameters
        ----------
        model_id : str
            HuggingFace repo id (used for config even when loading GGUF weights).
        gguf_path : str, optional
            Path to a local .gguf checkpoint. When provided, transformer weights
            come from this file; all other components still come from model_id.
        dtype : torch.dtype
            Compute dtype. bfloat16 recommended for FLUX on Ampere+.
        """
        self.model_id  = model_id
        self.gguf_path = gguf_path
        self.dtype     = dtype

    def activation_headroom_gb(
        self,
        height: int,
        width: int,
        batch_size: int = 1,
    ) -> float:
        """
        Estimate extra VRAM (GB) to reserve for activations beyond the base 0.8 GB
        """
        scale = (height * width) / (1024 * 1024) * batch_size
        if self.dtype == torch.float32:
            scale *= 2.0
        return max(0.4, self._empirical_extra_gb_1024 * scale)

    def load_transformer(self) -> nn.Module:
        """Load and return the transformer on CPU."""
        if self.gguf_path:
            return self._load_transformer_gguf()
        return self._load_transformer_hf()

    def _load_transformer_hf(self) -> nn.Module:
        raise NotImplementedError

    def _load_transformer_gguf(self) -> nn.Module:
        raise NotImplementedError

    def load_pipeline(self, transformer: nn.Module):
        """Return a diffusers pipeline with the given transformer."""
        raise NotImplementedError

    def __repr__(self) -> str:
        src = Path(self.gguf_path).name if self.gguf_path else self.model_id
        return (
            f"{self.__class__.__name__}("
            f"src={src!r}, "
            f"dtype={self.dtype}, "
            f"memory_usage_factor={self.memory_usage_factor})"
        )
