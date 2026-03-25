import gc

import torch


def vram_reserved_gb(device=None) -> float:
    """Bytes currently reserved by the PyTorch caching allocator, in GB."""
    torch.cuda.synchronize(device)
    return torch.cuda.memory_reserved(device) / 1024 ** 3


def vram_peak_gb(device=None) -> float:
    """Peak reserved VRAM since the last reset_peak() call, in GB."""
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_reserved(device) / 1024 ** 3


def reset_peak(device=None):
    """Reset PyTorch peak memory statistics."""
    torch.cuda.reset_peak_memory_stats(device)


def cleanup(device=None):
    """Run Python GC and release PyTorch's cached (but free) VRAM blocks."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
