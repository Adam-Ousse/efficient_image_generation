import time
import threading
import torch
import gc
import os

from .metrics import ResourceSnapshot, ResourceMetrics

# Import monitoring libraries
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not installed. Install with: pip install psutil")

try:
    import pynvml
    PYNVML_AVAILABLE = True
    pynvml.nvmlInit()
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: nvidia-ml-py not installed. Install with: pip install nvidia-ml-py")
except Exception:
    PYNVML_AVAILABLE = False


class ResourceMonitor:
    """
    Monitor system resources in background thread
    
    Usage:
        with ResourceMonitor() as monitor:
            # your code here
            pass
        metrics = monitor.get_metrics()
        metrics.print_summary()
        metrics.plot('usage.png')
    """
    
    def __init__(self, sample_rate_hz: float = 10.0, gpu_index: int = 0):
        """
        Args:
            sample_rate_hz: Sampling frequency (default 10 Hz)
            gpu_index: GPU device index
        """
        if not PSUTIL_AVAILABLE:
            raise ImportError("psutil required. Run: pip install psutil")
        
        self.sample_interval = 1.0 / sample_rate_hz
        self.gpu_index = gpu_index
        
        self._data = {
            "time": [],
            "vram_reserved_mb": [],
            "vram_allocated_mb": [],
            "vram_fragmentation_mb": [],
            "vram_total_mb": [],
            "ram_used_mb": [],
            "ram_total_mb": [],
            "gpu_util": [],
            "cpu_util": [],
            "power_watts": [],
            "pcie_tx_kb_s": [],
            "pcie_rx_kb_s": [],
        }
        
        self._thread = None
        self._running = [False]  # Use list so thread can see changes
        self._start_time = None
        self.process = psutil.Process()

        # Number of CPUs this job is allowed to use.
        # On SLURM: SLURM_CPUS_PER_TASK; otherwise all logical CPUs.
        self._n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', None) or
                           psutil.cpu_count(logical=True) or 1)
        
        # Initialize GPU if available
        self.gpu_handle = None
        if PYNVML_AVAILABLE:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except:
                pass
    
    def _sample(self):
        """Take one sample of all resources"""
        current_time = time.time() - self._start_time
        
        # CPU
        # self.process.cpu_percent() returns usage summed across all cores
        # used by THIS process (e.g. 800% if all 8 allocated cores are busy).
        # Divide by the number of CPUs allocated to this job so the result is
        # 0-100 % where 100 % means "all allocated CPUs are fully saturated".
        cpu_util = min(self.process.cpu_percent(interval=None) / self._n_cpus, 100.0)
        # RSS = physical RAM pages belonging to this process
        ram_used_mb  = self.process.memory_info().rss / (1024 ** 2)
        ram_total_mb = psutil.virtual_memory().total / (1024 ** 2)  # kept for reference
        
        # GPU metrics
        vram_reserved_mb  = 0
        vram_allocated_mb = 0
        vram_total_mb     = 0
        gpu_util    = 0
        power_watts = None
        pcie_tx_kb_s = None
        pcie_rx_kb_s = None
        
        if self.gpu_handle:
            try:
                # VRAM via pynvml (physical, real allocation)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                vram_reserved_mb  = mem_info.used / (1024 ** 2)  # total used by driver
                vram_total_mb     = mem_info.total / (1024 ** 2)
                
                # GPU utilization
                util_info = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_util = float(util_info.gpu)
                
                # Power
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                    power_watts = power_mw / 1000.0
                except Exception:
                    pass

                # PCIe throughput (KB/s)
                try:
                    pcie_tx_kb_s = pynvml.nvmlDeviceGetPcieThroughput(
                        self.gpu_handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
                    )
                    pcie_rx_kb_s = pynvml.nvmlDeviceGetPcieThroughput(
                        self.gpu_handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
                    )
                except Exception:
                    pass
            except Exception:
                pass
        
        # PyTorch allocator breakdown (independent of pynvml)
        if torch.cuda.is_available():
            try:
                vram_allocated_mb = torch.cuda.memory_allocated(self.gpu_index) / (1024 ** 2)
                if not self.gpu_handle:
                    # fallback if pynvml unavailable
                    vram_reserved_mb  = torch.cuda.memory_reserved(self.gpu_index) / (1024 ** 2)
                    vram_total_mb = (
                        torch.cuda.get_device_properties(self.gpu_index).total_memory / (1024 ** 2)
                    )
            except Exception:
                pass

        vram_fragmentation_mb = max(0.0, vram_reserved_mb - vram_allocated_mb)
        
        self._data["time"].append(current_time)
        self._data["vram_reserved_mb"].append(vram_reserved_mb)
        self._data["vram_allocated_mb"].append(vram_allocated_mb)
        self._data["vram_fragmentation_mb"].append(vram_fragmentation_mb)
        self._data["vram_total_mb"].append(vram_total_mb)
        self._data["ram_used_mb"].append(ram_used_mb)
        self._data["ram_total_mb"].append(ram_total_mb)
        self._data["gpu_util"].append(gpu_util)
        self._data["cpu_util"].append(cpu_util)
        self._data["power_watts"].append(power_watts)
        self._data["pcie_tx_kb_s"].append(pcie_tx_kb_s)
        self._data["pcie_rx_kb_s"].append(pcie_rx_kb_s)
    
    def _monitoring_loop(self):
        """Background thread loop"""
        while self._running[0]:
            self._sample()
            time.sleep(self.sample_interval)
    
    def start(self):
        """Start monitoring"""
        self._running[0] = True
        self._start_time = time.time()
        self.process.cpu_percent(interval=None)   # prime: first call always returns 0.0
        self._thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> ResourceMetrics:
        """Stop monitoring and return metrics"""
        self._running[0] = False
        if self._thread:
            self._thread.join(timeout=2.0)
        return self.get_metrics()
    
    def get_metrics(self) -> ResourceMetrics:
        """Compute and return metrics"""
        return ResourceMetrics(self._data, self._start_time)
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

def cleanup_gpu():
    """
    Proper VRAM release. Order matters.
    1. Break Python ref cycles first
    2. Force cyclic GC (handles __del__ chains)
    3. Release PyTorch's allocator cache
    4. Optionally trim the malloc arena (Linux only)
    """
    gc.collect()                          # first pass — marks cycle members
    gc.collect()                          # second pass — finalizes __del__
    gc.collect()                          # third pass — cleans up weakrefs etc.

    if torch.cuda.is_available():
        torch.cuda.synchronize()          # wait for all CUDA ops to finish
        torch.cuda.empty_cache()          # release PyTorch's cached blocks
        torch.cuda.ipc_collect()          # release IPC memory handles

    # Trim glibc malloc arena — recovers RAM too, Linux only
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass