"""
models/zimage.py — Tongyi-MAI Z-Image / Z-Image-Turbo descriptor.

Usage
-----
    m = ZImageModel("Tongyi-MAI/Z-Image-Turbo")
    pipe = m.load_pipeline().to("cuda")
    out  = pipe(prompt="a cat", num_inference_steps=4)
"""

from __future__ import annotations

import torch

from .base import ModelBase


class ZImageModel(ModelBase):
    """
    Descriptor for Tongyi-MAI Z-Image and Z-Image-Turbo.

    These models use ZImagePipeline directly (no separate transformer loading).
    GGUF quantisation is not supported for this architecture.
    """

    memory_usage_factor: float = 1.0
    vae_latent_factor:   int   = 8

    def __init__(self, model_id: str, dtype: torch.dtype = torch.bfloat16):
        super().__init__(model_id=model_id, gguf_path=None, dtype=dtype)

    def load_transformer(self):
        raise NotImplementedError(
            "ZImageModel does not expose a separate transformer. "
            "Use load_pipeline() directly."
        )

    def load_pipeline(self, transformer=None):
        from diffusers import ZImagePipeline
        print(f"  Loading ZImagePipeline: {self.model_id}")
        return ZImagePipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
