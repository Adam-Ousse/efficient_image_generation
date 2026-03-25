"""
offload.py — Budget-aware, module-granular GPU offloading.

Ported from ComfyUI:
  comfy/model_management.py  → pin_memory(), get_offload_stream(), cast_to()
  comfy/model_patcher.py     → _load_list(), load(), partially_load/unload()

Public API
----------
  SmartOffloadManager   — main class; attach to any nn.Module
  find_streaming_units  — enumerate coarsest non-overlapping streamable modules
  model_total_bytes     — total weight bytes in a model
  INFERENCE_HEADROOM_BYTES
  MAX_PINNED_BYTES
"""

from __future__ import annotations

import contextlib
import gc
import logging
import platform
import threading
import types
import weakref

import torch
import torch.nn as nn

# Optional GGUF support (diffusers ≥ 0.32)
try:
    from diffusers.quantizers.gguf.utils import (
        GGUFParameter,
        GGUFLinear,
        dequantize_gguf_tensor,
    )
    _HAS_GGUF = True
except ImportError:
    _HAS_GGUF = False
    GGUFParameter = None          # type: ignore[assignment]
    GGUFLinear = None             # type: ignore[assignment]
    dequantize_gguf_tensor = None # type: ignore[assignment]

logger = logging.getLogger(__name__)


# Constants

INFERENCE_HEADROOM_BYTES: int = int(0.8 * 1024 ** 3)   # 0.8 GB  (ComfyUI minimum_inference_memory)
DEFAULT_EXTRA_RESERVED:   int = int(0.4 * 1024 ** 3)   # 0.4 GB  safety margin
NUM_STREAMS: int = 2


def _cpu_total_bytes():
    try:
        import psutil
        return psutil.virtual_memory().total
    except ImportError:
        return 16 * 1024 ** 3


def _compute_max_pinned():
    """Per-OS pin limit. Mirrors ComfyUI MAX_PINNED_MEMORY."""
    if not torch.cuda.is_available():
        return -1
    ratio = 0.45 if platform.system() == "Windows" else 0.95
    cap = int(_cpu_total_bytes() * ratio)
    logger.info(f"PinnedMemory: max={cap // (1024**2)} MB ({ratio*100:.0f}% of RAM)")
    return cap


MAX_PINNED_BYTES: int = _compute_max_pinned()


# Memory helpers

def module_direct_bytes(module: nn.Module):
    """Bytes owned directly by this module (not by children). Mirrors ComfyUI module_size()."""
    total = 0
    for p in module.parameters(recurse=False):
        total += p.nbytes
    for b in module.buffers(recurse=False):
        total += b.nbytes
    return total


def module_subtree_bytes(module: nn.Module):
    """Total unique parameter + buffer bytes in the whole subtree."""
    seen = set()
    total = 0
    for p in module.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.nbytes
    for b in module.buffers():
        if id(b) not in seen:
            seen.add(id(b))
            total += b.nbytes
    return total


def model_total_bytes(model: nn.Module):
    return module_subtree_bytes(model)


# PinnedMemoryTracker
# Mirrors: ComfyUI PINNED_MEMORY dict + TOTAL_PINNED_MEMORY counter

class PinnedMemoryTracker:
    """
    Thread-safe registry of cudaHostRegister'd (page-locked) CPU tensors.

    Pinning eliminates the DMA bounce buffer: cudaHostRegister maps the
    existing tensor into CUDA's address space so H→D copies go direct
    (no intermediate copy, ~1.5-2× faster, and truly async with non_blocking=True).
    """

    def __init__(self):
        self._registry = {}   # data_ptr → nbytes
        self._total = 0
        self._lock = threading.Lock()

    @property
    def total_bytes(self):
        return self._total

    def pin(self, tensor: torch.Tensor):
        if MAX_PINNED_BYTES <= 0:
            return False
        if tensor.device.type != "cpu":
            return False
        if tensor.is_pinned():
            return False
        if not tensor.is_contiguous():
            return False
        ptr  = tensor.data_ptr()
        size = tensor.nbytes
        if ptr == 0 or size == 0:
            return False
        with self._lock:
            if ptr in self._registry:
                return False
            if self._total + size > MAX_PINNED_BYTES:
                return False
        # flag=0 (cudaHostRegisterDefault): pin for DMA only, do NOT map into
        # CUDA address space. flag=1 (Mapped) causes "resource already mapped"
        # errors when PyTorch async D→H copies touch already-registered memory.
        ret = torch.cuda.cudart().cudaHostRegister(ptr, size, 0)
        if ret == 0:
            with self._lock:
                self._registry[ptr] = size
                self._total += size
            return True
        else:
            logger.debug(f"cudaHostRegister failed ret={ret} size={size // 1024} KB")
            return False

    def unpin(self, tensor: torch.Tensor):
        if MAX_PINNED_BYTES <= 0:
            return False
        ptr  = tensor.data_ptr()
        size = tensor.nbytes
        with self._lock:
            registered_size = self._registry.pop(ptr, None)
        if registered_size is None:
            return False
        if size != registered_size:
            logger.warning(f"unpin: size mismatch (pinned={registered_size}, now={size})")
            with self._lock:
                self._registry[ptr] = registered_size
            return False
        ret = torch.cuda.cudart().cudaHostUnregister(ptr)
        if ret == 0:
            with self._lock:
                self._total -= registered_size
            return True
        else:
            logger.warning(f"cudaHostUnregister failed ret={ret}")
            return False

    def unpin_all(self):
        with self._lock:
            ptrs = list(self._registry.keys())
        for ptr in ptrs:
            try:
                torch.cuda.cudart().cudaHostUnregister(ptr)
            except Exception:
                pass
        with self._lock:
            self._registry.clear()
            self._total = 0
        logger.debug("All pinned memory released.")


# StreamPool
# Mirrors: ComfyUI STREAMS dict + stream_counters + get_offload_stream()

class StreamPool:
    """
    Round-robin pool of CUDA streams for async CPU→GPU weight transfers.

    Submitting H→D copies on a separate stream lets them run concurrently
    with GPU compute on the default stream — eliminating transfer stalls
    (this only works when source memory is pinned, see PinnedMemoryTracker).
    """

    def __init__(self, device: torch.device, num_streams: int = NUM_STREAMS):
        self.device = device
        self._streams = []
        self._counter = 0

        if num_streams > 0 and device.type == "cuda":
            for _ in range(num_streams):
                self._streams.append(torch.cuda.Stream(device=device, priority=0))
            logger.info(f"StreamPool: {num_streams} async transfer streams on {device}")
        else:
            logger.info(f"StreamPool: disabled (device={device}, num_streams={num_streams})")

    def next(self):
        """Return the next stream in round-robin order, synced to current compute."""
        if not self._streams:
            return None
        stream = self._streams[self._counter % len(self._streams)]
        self._counter += 1
        stream.wait_stream(torch.cuda.current_stream(self.device))
        return stream

    def sync_current_to(self, stream):
        """Make the compute stream wait for `stream`'s transfers to finish."""
        if stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(stream)

    @property
    def enabled(self):
        return len(self._streams) > 0


# Streaming unit discovery & VRAM budget
# Mirrors: ComfyUI model_patcher._load_list() + load_models_gpu()

# Unit: (name, module, subtree_bytes)

def find_streaming_units(model: nn.Module):
    """
    Find the coarsest non-overlapping modules that own memory, sorted largest-first.

    Rule: include module M if it has direct parameters/buffers. Remove any
    module whose name starts with an already-included prefix (i.e. it is a
    descendant of an included module). This keeps composite modules (e.g.
    MultiheadAttention which owns in_proj_weight AND has an out_proj child) as
    a single atomic streaming unit, preventing device-split forward errors.

    Returns a list of (name, module, subtree_bytes).
    """
    candidates = [
        (name, m, module_subtree_bytes(m))
        for name, m in model.named_modules()
        if module_direct_bytes(m) > 0
    ]
    # Sort by name length so parents come before their children
    candidates.sort(key=lambda x: len(x[0]))
    included_prefixes = []
    result = []
    for name, m, size in candidates:
        if not any(name.startswith(p + ".") for p in included_prefixes):
            included_prefixes.append(name)
            result.append((name, m, size))
    result.sort(key=lambda x: x[2], reverse=True)
    return result


def compute_weight_budget(
    device: torch.device,
    max_vram_bytes=None,
    extra_reserved_bytes: int = DEFAULT_EXTRA_RESERVED,
):
    """
    How many bytes of model weights can live on GPU?

    Uses memory_reserved() (not memory_allocated()) to account for the
    PyTorch caching allocator's freed-but-held blocks — same as ComfyUI.
    """
    headroom = INFERENCE_HEADROOM_BYTES + extra_reserved_bytes
    if device.type != "cuda":
        raise ValueError(f"Only CUDA devices supported, got {device}")
    torch.cuda.synchronize(device)
    already = torch.cuda.memory_reserved(device)
    free_cuda, _total = torch.cuda.mem_get_info(device)
    cached_reclaimable = already - torch.cuda.memory_allocated(device)
    free_vram = free_cuda + cached_reclaimable

    if max_vram_bytes is not None:
        budget = max(0, max_vram_bytes - already - headroom)
        logger.info(
            f"Budget (capped {max_vram_bytes//1024**2} MB): "
            f"already={already//1024**2} MB, headroom={headroom//1024**2} MB "
            f"→ {budget//1024**2} MB for weights"
        )
    else:
        budget = max(0, free_vram - headroom)
        logger.info(
            f"Budget (auto): free={free_vram//1024**2} MB, "
            f"headroom={headroom//1024**2} MB → {budget//1024**2} MB for weights"
        )
    return budget


def classify_modules(units, budget: int, num_streams: int = 2):
    """
    Greedy largest-first fill: split units into (resident, streaming).

    Resident modules go to GPU permanently; streaming modules stay on CPU
    and are moved with forward hooks. The lookahead buffer reserves room
    for concurrent DMA while computing, matching ComfyUI's offload_buffer logic.
    """
    resident  = []
    streaming = []
    mem_used = 0
    cast_buffer = 0

    for i, (name, module, size) in enumerate(units):
        lookahead = sum(
            units[j][2]
            for j in range(i + 1, min(i + 1 + num_streams, len(units)))
        )
        potential_buffer = max(cast_buffer, size + lookahead)
        fits = (mem_used + size + potential_buffer) <= budget

        if fits:
            resident.append((name, module, size))
            mem_used += size
        else:
            cast_buffer = potential_buffer
            streaming.append((name, module, size))

    logger.info(
        f"Classification → {len(resident)} resident "
        f"({sum(s for _,_,s in resident)//1024**2} MB on GPU), "
        f"{len(streaming)} streaming "
        f"({sum(s for _,_,s in streaming)//1024**2} MB paged)"
    )
    return resident, streaming


# SmartOffloadManager

class SmartOffloadManager:
    """
    Fit as much of `model` as possible into a VRAM budget.
    Stream the rest from CPU on demand, one module at a time.

    General fix (ring buffers)
    --------------------------
    The original implementation called mod.to(cpu) in the post-hook, which
    triggered a D→H copy and left the CUDA caching allocator's reserved pool
    inflated.  The ring-buffer approach eliminates this:

      • load(): pre-allocate GPU tensors ("ring buffers") for every streaming
        module's parameters.  CPU data is pinned once and never moved.
      • pre_hook: copy CPU weights → ring buffer (H→D), swap p.data pointers.
      • post_hook: restore p.data to CPU originals — zero bytes transferred.

    GGUF fix (dequant ring buffer)
    ------------------------------
    GGUFParameter weights are quantized bytes.  Without special handling the
    forward pass calls dequantize_gguf_tensor() which allocates a temporary
    FP16 tensor on the GPU for every GGUFLinear, causing further fragmentation.

    Instead we:
      • Exclude GGUFParameter from the general ring-buffer swap (they stay on
        CPU the whole time).
      • Allocate ONE shared FP16 ring buffer on the GPU sized to the largest
        GGUFLinear weight across all streaming modules.
      • Register per-GGUFLinear forward-pre/post hooks that:
          - dequantize the weight on the CPU (cheap, no GPU allocation)
          - H→D copy FP16 result into the single shared dequant ring buffer
          - temporarily replace the module's weight with a plain nn.Parameter
            wrapping the ring buffer so that GGUFLinear.forward_native()
            performs a regular F.linear call without any allocation.
      • Restore the original GGUFParameter in the post-hook.

    Parameters
    ----------
    model : nn.Module
        Model to manage. Must be on CPU when passed in.
    max_vram_gb : float, optional
        Hard cap on total VRAM this process should use.
        If None → uses all currently-free VRAM minus inference headroom.
    device : str | torch.device
        Target accelerator. Default: 'cuda'.
    num_streams : int
        Async DMA streams. 0 = synchronous, 2 = default (matches ComfyUI).
    extra_reserved_gb : float
        Additional headroom beyond the 0.8 GB inference reserve.
        Pass model.activation_headroom_gb(height, width) here.

    Usage
    -----
        mgr = SmartOffloadManager(model, max_vram_gb=6.0)
        mgr.load()
        output = model(input)   # hooks fire transparently
        mgr.unload()

        # Or as a context manager:
        with SmartOffloadManager(model, max_vram_gb=6.0):
            output = model(input)
    """

    def __init__(
        self,
        model: nn.Module,
        max_vram_gb=None,
        device: str | torch.device = "cuda",
        num_streams: int = 2,
        extra_reserved_gb: float = 0.4,
    ):
        self.model  = model
        self.device = torch.device(device)
        self.cpu    = torch.device("cpu")

        self._max_vram_bytes = (
            int(max_vram_gb * 1024 ** 3) if max_vram_gb is not None else None
        )
        self._extra = int(extra_reserved_gb * 1024 ** 3)

        self._pins    = PinnedMemoryTracker()
        self._streams = StreamPool(self.device, num_streams)

        self._resident  = []
        self._streaming = []
        self._hooks     = []
        self._loaded = False

        # Ring-buffer state (populated in load())
        # _ring_buffers[name] = list of entries, each either:
        #   ('param', param_obj, cpu_data_tensor, gpu_ring_tensor)
        #   ('buf',   submod,   bname, cpu_data_tensor, gpu_ring_tensor)
        # GGUFParameter weights are treated the same: their quantized bytes
        # (uint8) are copied H→D into a ring buffer; dequantization then runs
        # on GPU as usual, but no D→H copy ever occurs in post_hook.
        self._ring_buffers: dict[str, list] = {}
        # gguf
        self._patched_gguf_linears = [] 
        self._gguf_dq_buf = None
        self._gguf_layer_hooks = {}
        self._all_units = find_streaming_units(model)
        total = model_total_bytes(model)
        cap   = f"{max_vram_gb:.1f} GB cap" if max_vram_gb is not None else "auto cap"
        logger.info(
            f"SmartOffloadManager({model.__class__.__name__}, "
            f"{total//1024**2} MB total, {len(self._all_units)} units, {cap})"
        )


    def load(self, force_full_load: bool = False):
        """Classify modules, move residents to GPU, pin + hook streaming modules."""
        if self._loaded:
            logger.warning("Already loaded — skipping.")
            return self

        budget = compute_weight_budget(
            self.device,
            max_vram_bytes=self._max_vram_bytes,
            extra_reserved_bytes=self._extra,
        )

        total = model_total_bytes(self.model)
        if force_full_load or budget >= total:
            logger.info("Full load: entire model fits in VRAM budget.")
            self.model.to(self.device)
            self._resident  = list(self._all_units)
            self._streaming = []
            self._loaded    = True
            return self

        self._resident, self._streaming = classify_modules(
            self._all_units, budget, num_streams=len(self._streams._streams)
        )

        for _name, module, _size in self._resident:
            module.to(self.device, non_blocking=False)

        for _name, module, _size in self._streaming:
            module.to(self.cpu, non_blocking=False)

        for name, module, _size in self._streaming:
            self._build_ring_buffers(name, module)
            # Pin the CPU tensors once; they never move again.
            self._pin_module(module)
            h1, h2 = self._install_hooks(name, module)
            self._hooks.extend([h1, h2])
        self._gguf_dq_buf = self._alloc_gguf_dq_buf()
        if self._gguf_dq_buf is not None:
            for name, module, _size in self._streaming:
                self._setup_gguf_layer_hooks(name, module)
        self._loaded = True
        res_mb  = sum(s for _, _, s in self._resident)  // 1024 ** 2
        strm_mb = sum(s for _, _, s in self._streaming) // 1024 ** 2
        logger.info(
            f"Loaded: {res_mb} MB resident on GPU, "
            f"{strm_mb} MB streaming (lazy GPU alloc, one module at a time), "
            f"pinned={self._pins.total_bytes//1024**2} MB"
        )
        return self

    def unload(self):
        """Remove hooks, release pinned memory, move everything to CPU."""
        if not self._loaded:
            return self

        # Remove streaming-module hooks
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        for layer, orig_fwd, _ in self._patched_gguf_linears:
            layer.forward = orig_fwd
        self._patched_gguf_linears.clear()
        self._gguf_layer_hooks.clear()
        self._gguf_dq_buf = None
        # Free GPU ring buffers
        self._ring_buffers.clear()

        # Release pinned memory
        self._pins.unpin_all()

        # Move all weights back to CPU (safety net)
        self.model.to(self.cpu)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        gc.collect()
        self._resident  = []
        self._streaming = []
        self._loaded    = False
        logger.info("Unloaded: weights on CPU, hooks removed, pinned memory freed.")
        return self


    def _pin_module(self, module: nn.Module):
        for p in module.parameters():
            if p.device.type == "cpu":
                self._pins.pin(p.data)
        for b in module.buffers():
            if b.device.type == "cpu":
                self._pins.pin(b)

    def _unpin_module(self, module: nn.Module):
        for p in module.parameters():
            if p.device.type == "cpu":
                self._pins.unpin(p.data)
        for b in module.buffers():
            if b.device.type == "cpu":
                self._pins.unpin(b)


    def _is_gguf(self, t: torch.Tensor) -> bool:
        return _HAS_GGUF and isinstance(t, GGUFParameter)

    def _build_ring_buffers(self, name: str, module: nn.Module):
        """
        Store CPU tensor references only — no GPU pre-allocation.
        GPU tensors are allocated lazily in pre_hook (one module at a time)
        and freed immediately in post_hook with no D→H copy.
        This bounds VRAM to: resident + one streaming module, not all.
        """
        entries: list = []
        seen_ids: set = set()

        # Parameters
        for p in module.parameters():
            pid = id(p)
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            if p.device.type != "cpu":
                continue
            if self._is_gguf(p):
                continue
            entries.append(('param', p, p.data))          # no gpu_buf

        # Buffers (running_mean, running_var, etc.)
        for submod in module.modules():
            for bname, b in submod._buffers.items():
                if b is None:
                    continue
                bid = id(b)
                if bid in seen_ids:
                    continue
                seen_ids.add(bid)
                if b.device.type != "cpu":
                    continue
                entries.append(('buf', submod, bname, b))  # no gpu_buf

        self._ring_buffers[name] = entries

    def _alloc_gguf_dq_buf(self) -> torch.Tensor | None:
        """
        Allocate ONE shared FP16/BF16 ring buffer for GGUF dequantization,
        sized to the largest GGUFLinear weight (dequantized) across all
        streaming modules.  Returns None if there are no GGUFLinear layers.
        """
        max_numel = 0
        ref_dtype = torch.float16

        for _name, module, _size in self._streaming:
            for submod in module.modules():
                if not isinstance(submod, GGUFLinear):
                    continue
                w = submod.weight
                if not self._is_gguf(w):
                    continue
                numel = 1
                for d in w.quant_shape:
                    numel *= d
                if numel > max_numel:
                    max_numel = numel
                    # Use the layer's compute_dtype for the ring buffer
                    ref_dtype = getattr(submod, 'compute_dtype', torch.float16) or torch.float16

        if max_numel == 0:
            return None

        buf = torch.empty(max_numel, dtype=ref_dtype, device=self.device)
        logger.info(
            f"GGUF dequant ring buffer: {max_numel} elements "
            f"({buf.nbytes // 1024**2} MB, dtype={ref_dtype})"
        )
        return buf


    def _setup_gguf_layer_hooks(self, name: str, module: nn.Module):
        if self._gguf_dq_buf is None:
            return

        dq_buf = self._gguf_dq_buf
        mgr_ref = weakref.ref(self)

        for _lname, submod in module.named_modules():
            if not isinstance(submod, GGUFLinear):
                continue
            w = submod.weight
            if not self._is_gguf(w):
                continue

            # Capture dimensions
            out_f, in_f = w.quant_shape
            
            # Patch forward to use ring buffer when active
            orig_fwd = submod.forward
            def make_patched_fwd(layer, original_fwd):
                def fwd(inputs):
                    if getattr(layer, '_offload_ring_active', False):
                        return layer.forward_native(inputs)
                    return original_fwd(inputs)
                return fwd
            
            submod.forward = make_patched_fwd(submod, orig_fwd)
            
            # Pre-hook: CPU dequant → H→D copy → swap weight
            def make_pre_hook(layer, cpu_weight, o_f, i_f):
                def hook(mod, inp):
                    mgr = mgr_ref()
                    if mgr is None or not mgr._loaded:
                        return
                    
                    # CPU-side dequantize (cheap, no GPU alloc)
                    with torch.no_grad():
                        cpu_fp = dequantize_gguf_tensor(cpu_weight)
                        cdt = getattr(layer, 'compute_dtype', torch.float16) or torch.float16
                        cpu_fp = cpu_fp.to(cdt)
                    
                    # H→D into ring buffer slice
                    ring_slice = mgr._gguf_dq_buf[:o_f * i_f].view(o_f, i_f)
                    stream = mgr._streams.next()
                    
                    if stream is not None:
                        with torch.cuda.stream(stream):
                            ring_slice.copy_(cpu_fp, non_blocking=True)
                        mgr._streams.sync_current_to(stream)
                    else:
                        ring_slice.copy_(cpu_fp, non_blocking=False)
                    
                    # Swap to ring buffer parameter
                    layer._offload_saved_weight = layer._parameters['weight']
                    layer._parameters['weight'] = nn.Parameter(ring_slice, requires_grad=False)
                    layer._offload_ring_active = True
                return hook
            
            # Post-hook: restore GGUFParameter
            def make_post_hook(layer):
                def hook(mod, inp, out):
                    layer._offload_ring_active = False
                    saved = getattr(layer, '_offload_saved_weight', None)
                    if saved is not None:
                        layer._parameters['weight'] = saved
                        del layer._offload_saved_weight
                return hook
            
            h1 = submod.register_forward_pre_hook(make_pre_hook(submod, w, out_f, in_f))
            h2 = submod.register_forward_hook(make_post_hook(submod))
            self._hooks.extend([h1, h2])


    def _install_hooks(self, name: str, module: nn.Module):
        """
        PRE:  allocate GPU tensors, H→D copy via pinned memory, swap p.data.
        POST: restore p.data to CPU originals, delete GPU tensors.
              No D→H copy → no caching allocator inflation.
              Allocator reuses same-size freed blocks next step → minimal frag.
        """
        mgr_ref = weakref.ref(self)

        def pre_hook(mod: nn.Module, _inputs):
            mgr = mgr_ref()
            if mgr is None or not mgr._loaded:
                return
            entries = mgr._ring_buffers.get(name, [])
            stream  = mgr._streams.next()
            non_blocking = stream is not None
            ctx = torch.cuda.stream(stream) if non_blocking else contextlib.nullcontext()
            gpu_bufs = []
            with ctx:
                for e in entries:
                    if e[0] == 'param':
                        _tag, p, cpu_data = e
                        gpu_buf = torch.empty_like(cpu_data, device=mgr.device)
                        gpu_buf.copy_(cpu_data, non_blocking=non_blocking)
                        p.data = gpu_buf
                        gpu_bufs.append(gpu_buf)
                    else:  # 'buf'
                        _tag, submod, bname, cpu_data = e
                        gpu_buf = torch.empty_like(cpu_data, device=mgr.device)
                        gpu_buf.copy_(cpu_data, non_blocking=non_blocking)
                        submod._buffers[bname] = gpu_buf
                        gpu_bufs.append(gpu_buf)
            mgr._streams.sync_current_to(stream)
            mod._offload_gpu_bufs = gpu_bufs  # keep alive until post_hook

        def post_hook(mod: nn.Module, _inputs, _output):
            mgr = mgr_ref()
            if mgr is None or not mgr._loaded:
                return
            entries = mgr._ring_buffers.get(name, [])
            # Restore CPU data pointers — no D→H copy, no allocator inflation.
            for e in entries:
                if e[0] == 'param':
                    _tag, p, cpu_data = e
                    p.data = cpu_data
                else:  # 'buf'
                    _tag, submod, bname, cpu_data = e
                    submod._buffers[bname] = cpu_data
            # Drop GPU tensors. Caching allocator keeps the freed blocks and
            # reuses them for the next module (same size) → near-zero frag.
            if hasattr(mod, '_offload_gpu_bufs'):
                del mod._offload_gpu_bufs

        h1 = module.register_forward_pre_hook(pre_hook)
        h2 = module.register_forward_hook(post_hook)
        return h1, h2


    def __enter__(self):
        return self.load()

    def __exit__(self, *_):
        self.unload()


    def summary(self):
        res_mb  = sum(s for _, _, s in self._resident)  / 1024 ** 2
        strm_mb = sum(s for _, _, s in self._streaming) / 1024 ** 2
        free_mb = 0.0
        if self.device.type == "cuda":
            fc, _ = torch.cuda.mem_get_info(self.device)
            free_mb = fc / 1024 ** 2
        return {
            "loaded":       self._loaded,
            "resident_mb":  round(res_mb,  1),
            "streaming_mb": round(strm_mb, 1),
            "pinned_mb":    round(self._pins.total_bytes / 1024 ** 2, 1),
            "gpu_free_mb":  round(free_mb, 1),
            "num_hooks":    len(self._hooks),
            "num_units":    len(self._all_units),
        }

    def __repr__(self):
        s = self.summary()
        return (
            f"SmartOffloadManager("
            f"{self.model.__class__.__name__}, "
            f"loaded={s['loaded']}, "
            f"resident={s['resident_mb']} MB, "
            f"streaming={s['streaming_mb']} MB, "
            f"pinned={s['pinned_mb']} MB, "
            f"gpu_free={s['gpu_free_mb']} MB)"
        )
