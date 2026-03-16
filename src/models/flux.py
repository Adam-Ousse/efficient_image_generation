"""
models/flux.py — FLUX.1 / FLUX.2-klein model descriptor.
    Source: ComfyUI/comfy/supported_models.py, class Flux.

activation_headroom_gb() is overridden with an empirically measured formula
because ComfyUI's generic formula misses for FLUX by 4* (flash path) or 1.6*
(no-flash path). Measured on H100: ~4.6 GB activation overhead at 1024*1024.
Formula: extra = 3.8 GB * (H*W / 1024²) * batch  [*2 for fp32], min 0.4 GB.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .base import ModelBase


class FluxModel(ModelBase):
    """
    Descriptor for FLUX.1 and FLUX.2-klein models.

    Supports:
      - HuggingFace hub (bf16 / fp16)
      - Local GGUF checkpoint via diffusers GGUFQuantizationConfig

    Usage
    -----
        m = FluxModel("black-forest-labs/FLUX.2-klein-4B")
        transformer = m.load_transformer()

        m = FluxModel(
            model_id="black-forest-labs/FLUX.2-klein-4B",
            gguf_path="/path/to/flux-q4_k_m.gguf",
        )
        transformer = m.load_transformer()

        extra_gb = m.activation_headroom_gb(height=1024, width=1024)
        mgr = SmartOffloadManager(transformer, max_vram_gb=8.0,
                                  extra_reserved_gb=extra_gb)
    """
    _empirical_extra_gb_1024: float = 2329/1024 

    def activation_headroom_gb(
        self,
        height: int,
        width: int,
        batch_size: int = 1,
    ) -> float:
        """
        Activation memory (GB) to reserve on top of the base 0.8 GB.

        Uses a linear fit calibrated on real measurements rather than
        ComfyUI's generic formula (which is off by 4* for FLUX).

            extra = 3.8 GB * (H * W / 1024²) * batch  [* 2 for fp32]
            clamped to minimum 0.4 GB
        """
        scale = (height * width) / (1024 * 1024) * batch_size
        if self.dtype == torch.float32:
            scale *= 2.0
        return max(0.4, self._empirical_extra_gb_1024 * scale)

    def _load_transformer_hf(self):
        from diffusers import Flux2Transformer2DModel
        print(f"  Loading transformer from HuggingFace: {self.model_id}")
        return Flux2Transformer2DModel.from_pretrained(
            self.model_id,
            subfolder="transformer",
            torch_dtype=self.dtype,
        )

    def _resolve_gguf_path(self) -> str:
        """Return a local filesystem path, downloading from HF hub if needed."""
        if Path(self.gguf_path).exists():
            return self.gguf_path

        parts = self.gguf_path.split("/")
        if len(parts) >= 3:
            from huggingface_hub import hf_hub_download
            repo_id  = f"{parts[0]}/{parts[1]}"
            filename = "/".join(parts[2:])
            print(f"  Resolving GGUF from HuggingFace: {repo_id} / {filename}")
            try:
                local = hf_hub_download(repo_id=repo_id, filename=filename,
                                        local_files_only=True)
                print(f"  ✓ Found in cache: {local}")
                return local
            except Exception:
                print("  Downloading from HuggingFace…")
                local = hf_hub_download(repo_id=repo_id, filename=filename)
                print(f"  ✓ Downloaded to: {local}")
                return local

        raise ValueError(
            f"gguf_path '{self.gguf_path}' is neither a local file nor a valid "
            "'owner/repo/filename.gguf' HuggingFace path."
        )

    def _load_transformer_gguf(self):
        from diffusers import Flux2Transformer2DModel, GGUFQuantizationConfig
        local_path = self._resolve_gguf_path()
        print(f"  Loading GGUF transformer: {Path(local_path).name}")
        return Flux2Transformer2DModel.from_single_file(
            local_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype),
            torch_dtype=self.dtype,
            config=self.model_id,
            subfolder="transformer",
        )

    def load_pipeline(self, transformer=None):
        from diffusers import Flux2KleinPipeline
        if transformer is None:
            transformer = self.load_transformer()
        return Flux2KleinPipeline.from_pretrained(
            self.model_id,
            transformer=transformer,
            torch_dtype=self.dtype,
        )
