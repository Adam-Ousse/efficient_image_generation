"""
offload — smart GPU memory offloading ported from ComfyUI.

Public API
----------
    from offload import SmartOffloadManager

    mgr = SmartOffloadManager(model, max_vram_gb=6.0)
    mgr.load()
    output = model(input)
    mgr.unload()

    # or:
    with SmartOffloadManager(model, max_vram_gb=6.0):
        output = model(input)
"""

# from .manager import SmartOffloadManager
# from .memory import PinnedMemoryTracker, MAX_PINNED_BYTES
# from .models import ModelBase, FluxModel
# from .modules import (
#     find_streaming_units,
#     classify_modules,
#     compute_weight_budget,
#     model_total_bytes,
#     module_subtree_bytes,
#     INFERENCE_HEADROOM_BYTES,
# )
# from .streams import StreamPool, NUM_STREAMS

from .offload import (
    SmartOffloadManager,
    PinnedMemoryTracker,
    MAX_PINNED_BYTES,
    find_streaming_units,
    classify_modules,
    compute_weight_budget,
    model_total_bytes,
    module_subtree_bytes,
    INFERENCE_HEADROOM_BYTES,
    StreamPool,
    NUM_STREAMS,
)
from .pipeline_utils import fix_execution_device, fix_cpu_text_encoder, run_generation
__all__ = [
    # Main class
    "SmartOffloadManager",
    # # Model descriptors
    # Utilities
    "find_streaming_units",
    "classify_modules",
    "compute_weight_budget",
    "model_total_bytes",
    "module_subtree_bytes",
    # Internals (for experiments / custom builds)
    "PinnedMemoryTracker",
    "StreamPool",
    # Constants
    "NUM_STREAMS",
    "INFERENCE_HEADROOM_BYTES",
    "MAX_PINNED_BYTES",
    # Pipeline utils
    "fix_execution_device",
    "fix_cpu_text_encoder",
    "run_generation",
]
