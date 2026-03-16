import time
import pandas as pd


class ResourceSnapshot:
    """Single measurement (not used in simplified version, kept for compatibility)"""
    pass


class ResourceMetrics:
    """Resource usage metrics with export and plotting"""
    
    def __init__(self, data, start_time):
        """
        Args:
            data: Dict with lists of measurements
            start_time: When monitoring started
        """
        self.data = data
        self.start_time = start_time
        self.duration_seconds = data["time"][-1] if data["time"] else 0
        self.num_samples = len(data["time"])
        
        # Compute statistics
        if self.num_samples > 0:
            self.vram_reserved_mean_mb  = sum(data["vram_reserved_mb"]) / self.num_samples
            self.vram_reserved_max_mb   = max(data["vram_reserved_mb"])

            self.vram_allocated_mean_mb = sum(data["vram_allocated_mb"]) / self.num_samples
            self.vram_allocated_max_mb  = max(data["vram_allocated_mb"])

            self.vram_frag_mean_mb      = sum(data["vram_fragmentation_mb"]) / self.num_samples
            self.vram_frag_max_mb       = max(data["vram_fragmentation_mb"])

            self.ram_mean_mb = sum(data["ram_used_mb"]) / self.num_samples
            self.ram_max_mb  = max(data["ram_used_mb"])
            self.ram_min_mb  = min(data["ram_used_mb"])
            
            self.gpu_util_mean = sum(data["gpu_util"]) / self.num_samples
            self.gpu_util_max  = max(data["gpu_util"])
            
            self.cpu_util_mean = sum(data["cpu_util"]) / self.num_samples
            self.cpu_util_max  = max(data["cpu_util"])
            
            # Power (optional)
            power_values = [p for p in data["power_watts"] if p is not None]
            if power_values:
                self.power_mean_watts  = sum(power_values) / len(power_values)
                self.power_max_watts   = max(power_values)
                self.power_total_joules = self.power_mean_watts * self.duration_seconds
            else:
                self.power_mean_watts = self.power_max_watts = self.power_total_joules = None

            # PCIe (optional)
            tx_vals = [v for v in data["pcie_tx_kb_s"] if v is not None]
            rx_vals = [v for v in data["pcie_rx_kb_s"] if v is not None]
            self.pcie_tx_mean_kb_s = sum(tx_vals) / len(tx_vals) if tx_vals else None
            self.pcie_tx_max_kb_s  = max(tx_vals) if tx_vals else None
            self.pcie_rx_mean_kb_s = sum(rx_vals) / len(rx_vals) if rx_vals else None
            self.pcie_rx_max_kb_s  = max(rx_vals) if rx_vals else None
        else:
            self.vram_reserved_mean_mb = self.vram_reserved_max_mb = 0
            self.vram_allocated_mean_mb = self.vram_allocated_max_mb = 0
            self.vram_frag_mean_mb = self.vram_frag_max_mb = 0
            self.ram_mean_mb = self.ram_max_mb = self.ram_min_mb = 0
            self.gpu_util_mean = self.gpu_util_max = 0
            self.cpu_util_mean = self.cpu_util_max = 0
            self.power_mean_watts = self.power_max_watts = self.power_total_joules = None
            self.pcie_tx_mean_kb_s = self.pcie_tx_max_kb_s = None
            self.pcie_rx_mean_kb_s = self.pcie_rx_max_kb_s = None

        # Legacy alias so old code that reads vram_mean_mb / vram_max_mb still works
        self.vram_mean_mb = self.vram_reserved_mean_mb
        self.vram_max_mb  = self.vram_reserved_max_mb
        self.vram_min_mb  = min(data["vram_reserved_mb"]) if self.num_samples > 0 else 0
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        df = pd.DataFrame(self.data)
        if len(df) > 0:
            df = df.set_index('time')
        return df
    
    def save_csv(self, filepath: str):
        """Save to CSV"""
        df = self.to_dataframe()
        df.to_csv(filepath)
        print(f"Saved metrics to: {filepath}")
    
    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "=" * 60)
        print("RESOURCE USAGE SUMMARY")
        print("=" * 60)
        print(f"Duration: {self.duration_seconds:.2f}s")
        print(f"Samples:  {self.num_samples} ({self.num_samples/self.duration_seconds:.1f} Hz)")
        print()
        print(f"VRAM reserved  : Mean {self.vram_reserved_mean_mb:7.1f} MB  | Peak {self.vram_reserved_max_mb:7.1f} MB")
        print(f"VRAM allocated : Mean {self.vram_allocated_mean_mb:7.1f} MB  | Peak {self.vram_allocated_max_mb:7.1f} MB")
        print(f"VRAM fragment  : Mean {self.vram_frag_mean_mb:7.1f} MB  | Peak {self.vram_frag_max_mb:7.1f} MB  (reserved - allocated)")
        print(f"RAM (proc RSS) : Mean {self.ram_mean_mb:7.1f} MB  | Peak {self.ram_max_mb:7.1f} MB")
        print(f"GPU util       : Mean {self.gpu_util_mean:5.1f}%    | Peak {self.gpu_util_max:5.1f}%")
        print(f"CPU util       : Mean {self.cpu_util_mean:5.1f}%    | Peak {self.cpu_util_max:5.1f}%")
        
        if self.power_mean_watts:
            print(f"Power          : Mean {self.power_mean_watts:5.1f} W   | Peak {self.power_max_watts:.1f} W")
            print(f"Energy         : {self.power_total_joules:.1f} J ({self.power_total_joules/3600:.4f} Wh)")

        if self.pcie_tx_mean_kb_s is not None:
            print(f"PCIe TX        : Mean {self.pcie_tx_mean_kb_s/1024:6.1f} MB/s | Peak {self.pcie_tx_max_kb_s/1024:.1f} MB/s")
            print(f"PCIe RX        : Mean {self.pcie_rx_mean_kb_s/1024:6.1f} MB/s | Peak {self.pcie_rx_max_kb_s/1024:.1f} MB/s")
        
        print("=" * 60)
    
    def plot(self, save_path: str = None):
        """Plot resource usage"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return
        
        df = self.to_dataframe()
        if df.empty:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        fig.suptitle('Resource Usage Over Time', fontsize=16, fontweight='bold')
        
        # VRAM breakdown
        ax = axes[0, 0]
        ax.plot(df.index, df['vram_reserved_mb'],  color='red',    linewidth=2, label='reserved')
        ax.plot(df.index, df['vram_allocated_mb'], color='orange', linewidth=2, label='allocated')
        ax.fill_between(df.index, df['vram_allocated_mb'], df['vram_reserved_mb'],
                        alpha=0.25, color='red', label='fragmented')
        ax.fill_between(df.index, 0, df['vram_allocated_mb'], alpha=0.3, color='orange')
        ax.set_ylabel('VRAM (MB)')
        ax.set_title(f'GPU Memory (peak reserved {self.vram_reserved_max_mb:.0f} MB)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # VRAM fragmentation
        ax = axes[0, 1]
        ax.plot(df.index, df['vram_fragmentation_mb'], color='crimson', linewidth=2)
        ax.fill_between(df.index, 0, df['vram_fragmentation_mb'], alpha=0.3, color='crimson')
        ax.set_ylabel('VRAM (MB)')
        ax.set_title(f'Fragmentation = reserved - allocated (peak {self.vram_frag_max_mb:.0f} MB)')
        ax.grid(True, alpha=0.3)
        
        # RAM
        ax = axes[1, 0]
        ax.plot(df.index, df['ram_used_mb'], color='blue', linewidth=2)
        ax.fill_between(df.index, 0, df['ram_used_mb'], alpha=0.3, color='blue')
        ax.set_ylabel('RAM (MB)')
        ax.set_title(f'Process RAM / RSS (Peak: {self.ram_max_mb:.0f} MB)')
        ax.grid(True, alpha=0.3)
        
        # GPU utilization
        ax = axes[1, 1]
        ax.plot(df.index, df['gpu_util'], color='green', linewidth=2)
        ax.fill_between(df.index, 0, df['gpu_util'], alpha=0.3, color='green')
        ax.set_ylabel('GPU Utilization (%)')
        ax.set_title(f'GPU Usage (Peak: {self.gpu_util_max:.0f}%)')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

        # PCIe throughput
        ax = axes[2, 0]
        if 'pcie_tx_kb_s' in df.columns and df['pcie_tx_kb_s'].notna().any():
            tx_mbs = df['pcie_tx_kb_s'].fillna(0) / 1024
            rx_mbs = df['pcie_rx_kb_s'].fillna(0) / 1024
            ax.plot(df.index, tx_mbs, color='purple',   linewidth=2, label='TX')
            ax.plot(df.index, rx_mbs, color='steelblue', linewidth=2, label='RX')
            ax.set_ylabel('PCIe (MB/s)')
            ax.set_title('PCIe Throughput')
            ax.legend(fontsize=8)
        else:
            ax.set_title('PCIe Throughput (unavailable)')
        ax.set_xlabel('Time (seconds)')
        ax.grid(True, alpha=0.3)

        # CPU
        ax = axes[2, 1]
        ax.plot(df.index, df['cpu_util'], color='orange', linewidth=2)
        ax.fill_between(df.index, 0, df['cpu_util'], alpha=0.3, color='orange')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('CPU Utilization (%)')
        ax.set_title(f'CPU Usage (Peak: {self.cpu_util_max:.0f}%)')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        else:
            plt.show()
        
        plt.close()

