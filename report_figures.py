"""Generate figures for report_benchmark.tex from summary.csv."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
CSV  = Path("results/offload_benchmark_final/summary.csv")
OUT  = Path("figures")
OUT.mkdir(exist_ok=True)

BUDGET_GB = 6.0
MB_TO_GB  = 1 / 1024
KB_TO_MB  = 1 / 1024

# Strategies shown in every figure (no smart variants)
STRATS     = ['none', 'cpu_offload', 'sequential', 'group_offload', "cpu_only"]
STRAT_COLS = {
    'none':         '#4878cf',
    'cpu_offload':  '#ee854a',
    'sequential':   '#6acc65',
    'group_offload':'#d65f5f',
    'cpu_only':     '#956cb4', 
}
STRAT_LABEL = {
    'none':         'Full GPU',
    'cpu_offload':  'CPU Offload',
    'sequential':   'Sequential\n(FP16 only)',
    'group_offload':'Group Offload',
    'cpu_only':     'CPU Only',
}

QUANTS = ['FP16', 'Q2_K', 'Q3_K_M', 'Q4_K_M', 'Q5_K_M']
QUANT_MODEL = {
    'FP16':     'FLUX2-Klein-FP16',
    'Q2_K':     'FLUX2-Klein-Q2_K',
    'Q3_K_M':   'FLUX2-Klein-Q3_K_M',
    'Q4_K_M':   'FLUX2-Klein-Q4_K_M',
    'Q5_K_M':   'FLUX2-Klein-Q5_K_M',
}
QUANT_KEYS  = ['FP16', 'Q2_K', 'Q3_K_M', 'Q4_K_M', 'Q5_K_M']
QUANT_TICK  = ['FP16', 'Q2_K', 'Q3_K_M', 'Q4_K_M', 'Q5_K_M']

plt.rcParams.update({
    'font.family': 'serif',
    'font.size':   10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':      True,
    'axes.grid.axis': 'y',
    'grid.alpha':     0.35,
    'figure.dpi':     150,
})

df = pd.read_csv(CSV)

def row(model_key, offload):
    """Return summary row or None if not present."""
    model = QUANT_MODEL.get(model_key, model_key)
    r = df[(df['model'] == model) & (df['offload'] == offload)]
    return r.iloc[0] if len(r) else None

def row_by_run_id(run_id):
    """Return summary row by exact run_id or None if not present."""
    r = df[df['run_id'] == run_id]
    return r.iloc[0] if len(r) else None

def fp16(offload):
    return row('FP16', offload)


# Fig 1 – VRAM: reserved vs allocated, FP16 strategies (no smart)
fig, ax = plt.subplots(figsize=(7.0, 3.8))
x = np.arange(len(STRATS))
w = 0.32

res_v   = [fp16(s)['vram_reserved_peak_mb_mean']  * MB_TO_GB for s in STRATS]
alloc_v = [fp16(s)['vram_allocated_peak_mb_mean'] * MB_TO_GB for s in STRATS]

comfy_row = row_by_run_id('FLUX2-Klein-FP16__ComfyUI_lowvram6GB')
if comfy_row is not None:
    vram_strats = STRATS + ['comfyui_lowvram']
    vram_labels = {**STRAT_LABEL, 'comfyui_lowvram': 'ComfyUI\nlowvram 6GB'}
    x_vram = np.arange(len(vram_strats))
    res_v.append(comfy_row['vram_reserved_peak_mb_mean'] * MB_TO_GB)
    alloc_v.append(comfy_row['vram_allocated_peak_mb_mean'] * MB_TO_GB)
else:
    vram_strats = STRATS
    vram_labels = STRAT_LABEL
    x_vram = x

ax.bar(x_vram - w/2, res_v,   w, label='Reserved',  color='#4878cf', alpha=0.85)
ax.bar(x_vram + w/2, alloc_v, w, label='Allocated', color='#4878cf', alpha=0.38,
       hatch='//', edgecolor='#4878cf')

for xi, r, a in zip(x_vram, res_v, alloc_v):
    ax.annotate(f'+{r-a:.1f}', xy=(xi - w/2, r), xytext=(0, 3),
                textcoords='offset points', ha='center', fontsize=7.5, color='#444')

ax.axhline(BUDGET_GB, color='crimson', linestyle='--', linewidth=1.3,
           label=f'{BUDGET_GB:.0f} GB budget')
ax.set_xticks(x_vram)
ax.set_xticklabels([vram_labels[s] for s in vram_strats], rotation=12, ha='right')
ax.set_ylabel('VRAM (GB)')
ax.set_title('Peak VRAM — FP16 FLUX.2-Klein-4B', fontweight='bold', pad=8)
ax.legend(fontsize=9, framealpha=0.9)
ax.set_ylim(0, 24)
fig.tight_layout()
# fig.savefig(OUT / 'fig_vram_fp16.pdf', bbox_inches='tight')
fig.savefig(OUT / 'fig_vram_fp16.png', bbox_inches='tight')
plt.close(fig)
print("✓ fig_vram_fp16")


# Fig 2 – Timing stacked bar: load / gen / overhead, FP16 (no smart)
fig, ax = plt.subplots(figsize=(7.0, 3.8))
load_v  = [fp16(s)['load_time_s_mean']  for s in STRATS]
gen_v   = [fp16(s)['gen_time_s_mean']   for s in STRATS]
total_v = [fp16(s)['total_time_s_mean'] for s in STRATS]
over_v  = [t - l - g for t, l, g in zip(total_v, load_v, gen_v)]

bw = 0.5
ax.bar(x, load_v, bw, label='Load',     color='#4878cf', alpha=0.85)
ax.bar(x, gen_v,  bw, label='Generate', color='#ee854a', alpha=0.85, bottom=load_v)
ax.bar(x, over_v, bw, label='Overhead', color='#6acc65', alpha=0.70,
       bottom=[l+g for l,g in zip(load_v, gen_v)])

ax.set_xticks(x)
ax.set_xticklabels([STRAT_LABEL[s] for s in STRATS], rotation=12, ha='right')
ax.set_ylabel('Time (s)')
ax.set_title('End-to-end Latency Breakdown — FP16', fontweight='bold', pad=8)
ax.legend(fontsize=9, framealpha=0.9)
fig.tight_layout()
# fig.savefig(OUT / 'fig_timing_fp16.pdf', bbox_inches='tight')
fig.savefig(OUT / 'fig_timing_fp16.png', bbox_inches='tight')
plt.close(fig)
print("✓ fig_timing_fp16")


# Fig 3 – PCIe bandwidth (mean + peak), FP16 (no smart)
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))

tx_m = [fp16(s)['pcie_tx_mean_kb_s_mean'] * KB_TO_MB for s in STRATS]
rx_m = [fp16(s)['pcie_rx_mean_kb_s_mean'] * KB_TO_MB for s in STRATS]
tx_p = [fp16(s)['pcie_tx_max_kb_s_mean']  * KB_TO_MB for s in STRATS]
rx_p = [fp16(s)['pcie_rx_max_kb_s_mean']  * KB_TO_MB for s in STRATS]

for ax, tv, rv, title in [
    (axes[0], tx_m, rx_m, 'Mean PCIe Bandwidth'),
    (axes[1], tx_p, rx_p, 'Peak PCIe Bandwidth'),
]:
    ax.bar(x - w/2, tv, w, label='TX  GPU→CPU', color='#4878cf', alpha=0.85)
    ax.bar(x + w/2, rv, w, label='RX  CPU→GPU', color='#ee854a', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([STRAT_LABEL[s] for s in STRATS], rotation=12, ha='right')
    ax.set_ylabel('MB/s')
    ax.set_title(title, fontweight='bold', pad=8)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.35)

fig.suptitle('PCIe Bandwidth — FP16 FLUX.2-Klein-4B', fontweight='bold', y=1.02)
fig.tight_layout()
# fig.savefig(OUT / 'fig_pcie_fp16.pdf', bbox_inches='tight')
fig.savefig(OUT / 'fig_pcie_fp16.png', bbox_inches='tight')
plt.close(fig)
print("✓ fig_pcie_fp16")


# Fig 4 – Quantization effect: 3 panels × strategies
#   Panel A: Reserved VRAM
#   Panel B: Generation time
#   Panel C: Load time
#   Strategies: Full GPU, CPU Offload, Group Offload, CPU Only
Q_STRATS = ['none', 'cpu_offload', 'group_offload', 'cpu_only']
Q_COLORS = [STRAT_COLS[s] for s in Q_STRATS]
Q_LABELS_SHORT = ['Full GPU', 'CPU Offload', 'Group Offload', 'CPU Only']

xq  = np.arange(len(QUANT_KEYS))
wq  = 0.18
off = np.linspace(-1.5 * wq, 1.5 * wq, len(Q_STRATS))

def quant_vals(metric, offload):
    vals = []
    for qk in QUANT_KEYS:
        r = row(qk, offload)
        vals.append(r[metric] if r is not None else float('nan'))
    return vals

fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))

ax = axes[0]
for oi, s, c, lbl in zip(off, Q_STRATS, Q_COLORS, Q_LABELS_SHORT):
    vals = [v * MB_TO_GB for v in quant_vals('vram_reserved_peak_mb_mean', s)]
    ax.bar(xq + oi, vals, wq, color=c, alpha=0.85, label=lbl)
ax.axhline(BUDGET_GB, color='crimson', linestyle='--', linewidth=1.2,
           label=f'{BUDGET_GB:.0f} GB budget')
ax.set_xticks(xq); ax.set_xticklabels(QUANT_TICK, rotation=15, ha='right')
ax.set_ylabel('VRAM reserved (GB)')
ax.set_title('Reserved VRAM', fontweight='bold', pad=8)
ax.legend(fontsize=8, framealpha=0.9)

ax = axes[1]
for oi, s, c, lbl in zip(off, Q_STRATS, Q_COLORS, Q_LABELS_SHORT):
    vals = quant_vals('gen_time_s_mean', s)
    ax.bar(xq + oi, vals, wq, color=c, alpha=0.85, label=lbl)
ax.set_xticks(xq); ax.set_xticklabels(QUANT_TICK, rotation=15, ha='right')
ax.set_ylabel('Time (s)')
ax.set_title('Generation Time', fontweight='bold', pad=8)
ax.legend(fontsize=8, framealpha=0.9)

ax = axes[2]
for oi, s, c, lbl in zip(off, Q_STRATS, Q_COLORS, Q_LABELS_SHORT):
    vals = quant_vals('load_time_s_mean', s)
    ax.bar(xq + oi, vals, wq, color=c, alpha=0.85, label=lbl)
ax.set_xticks(xq); ax.set_xticklabels(QUANT_TICK, rotation=15, ha='right')
ax.set_ylabel('Time (s)')
ax.set_title('Load Time', fontweight='bold', pad=8)
ax.legend(fontsize=8, framealpha=0.9)

# Shared legend below all 3 panels instead of per-panel (avoids overlap)
for ax in axes:
    ax.get_legend().remove()
handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.85) for c in Q_COLORS]
fig.legend(handles, Q_LABELS_SHORT, loc='lower center', ncol=4,
           fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.08))
fig.suptitle('Effect of Quantization Across Offloading Strategies', fontweight='bold', y=1.02)
fig.tight_layout()
# fig.savefig(OUT / 'fig_quant.pdf', bbox_inches='tight')
fig.savefig(OUT / 'fig_quant.png', bbox_inches='tight')
plt.close(fig)
print("✓ fig_quant")


# Fig 5 – Power draw and energy consumption, FP16 (no smart)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.8))
x4 = np.arange(len(STRATS))

pow_mean = [fp16(s)['power_mean_w_mean'] for s in STRATS]
pow_max  = [fp16(s)['power_max_w_mean']  for s in STRATS]
total_t  = [fp16(s)['total_time_s_mean'] for s in STRATS]
energy_wh = [p * t / 3600 for p, t in zip(pow_mean, total_t)]  # Wh

ax1.bar(x4 - w/2, pow_mean, w, label='Mean power', color='#e05c5c', alpha=0.85)
ax1.bar(x4 + w/2, pow_max,  w, label='Peak power', color='#e05c5c', alpha=0.40,
        hatch='//', edgecolor='#e05c5c')
ax1.set_xticks(x4); ax1.set_xticklabels([STRAT_LABEL[s] for s in STRATS], rotation=12, ha='right')
ax1.set_ylabel('Power (W)'); ax1.set_title('GPU Power Draw', fontweight='bold', pad=8)
ax1.legend(fontsize=9, framealpha=0.9)

colors4 = [STRAT_COLS[s] for s in STRATS]
ax2.bar(x4, energy_wh, 0.5, color=colors4, alpha=0.85)
for xi, e in zip(x4, energy_wh):
    ax2.annotate(f'{e:.2f}', xy=(xi, e), xytext=(0, 3),
                 textcoords='offset points', ha='center', fontsize=8, color='#333')
ax2.set_xticks(x4); ax2.set_xticklabels([STRAT_LABEL[s] for s in STRATS], rotation=12, ha='right')
ax2.set_ylabel('Energy (Wh)'); ax2.set_title('Total Energy per Image', fontweight='bold', pad=8)

fig.suptitle('Power and Energy — FP16 FLUX.2-Klein-4B', fontweight='bold', y=1.02)
fig.tight_layout()
# fig.savefig(OUT / 'fig_power_fp16.pdf', bbox_inches='tight')
fig.savefig(OUT / 'fig_power_fp16.png', bbox_inches='tight')
plt.close(fig)
print("✓ fig_power_fp16")


# Fig 6 – System RAM usage: FP16 strategies (left) + quant effect (right)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.8))

# Panel A: FP16 RAM peak per strategy
ram_fp16 = [fp16(s)['ram_peak_mb_mean'] / 1024 for s in STRATS]
lcolors  = [STRAT_COLS[s] for s in STRATS]
ax1.bar(x4, ram_fp16, 0.5, color=lcolors, alpha=0.85)
for xi, v in zip(x4, ram_fp16):
    ax1.annotate(f'{v:.1f}', xy=(xi, v), xytext=(0, 3),
                 textcoords='offset points', ha='center', fontsize=8)
ax1.axhline(32, color='crimson', linestyle='--', linewidth=1.2, label='32 GB budget')
ax1.set_xticks(x4); ax1.set_xticklabels([STRAT_LABEL[s] for s in STRATS], rotation=12, ha='right')
ax1.set_ylabel('RAM (GB)'); ax1.set_title('Peak System RAM — FP16', fontweight='bold', pad=8)
ax1.legend(fontsize=9, framealpha=0.9)

# Panel B: RAM peak per quant, strategies
xq2 = np.arange(len(QUANT_KEYS)); wq2 = 0.18; off2 = np.linspace(-1.5 * wq2, 1.5 * wq2, len(Q_STRATS))
Q_STRATS2 = ['none', 'cpu_offload', 'group_offload', 'cpu_only']
Q_COLS2   = [STRAT_COLS[s] for s in Q_STRATS2]
Q_LBLS2   = ['Full GPU', 'CPU Offload', 'Group Offload', 'CPU Only']
for oi, s, c, lbl in zip(off2, Q_STRATS2, Q_COLS2, Q_LBLS2):
    vals = [float(row(qk, s)['ram_peak_mb_mean']) / 1024 if row(qk, s) is not None else float('nan')
            for qk in QUANT_KEYS]
    ax2.bar(xq2 + oi, vals, wq2, color=c, alpha=0.85, label=lbl)
ax2.axhline(32, color='crimson', linestyle='--', linewidth=1.2, label='32 GB budget')
ax2.set_xticks(xq2); ax2.set_xticklabels(QUANT_TICK, rotation=15, ha='right')
ax2.set_ylabel('RAM (GB)'); ax2.set_title('Peak System RAM vs Quantization', fontweight='bold', pad=8)
ax2.legend(fontsize=8, framealpha=0.9, loc='upper right')

fig.suptitle('System RAM Usage', fontweight='bold', y=1.02)
fig.tight_layout()
# fig.savefig(OUT / 'fig_ram.pdf', bbox_inches='tight')
fig.savefig(OUT / 'fig_ram.png', bbox_inches='tight')
plt.close(fig)
print("✓ fig_ram")


# Fig 7 – Power and Energy vs Quantization
#   Panel A: Mean power (W)
#   Panel B: Peak power (W)
#   Panel C: Energy per image (Wh = power_mean × total_time / 3600)
#   Strategies: Full GPU, CPU Offload, Group Offload, CPU Only
fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))

for oi, s, c in zip(off, Q_STRATS, Q_COLORS):
    axes[0].bar(xq + oi, quant_vals('power_mean_w_mean', s), wq, color=c, alpha=0.85)
    axes[1].bar(xq + oi, quant_vals('power_max_w_mean', s),  wq, color=c, alpha=0.85)
    p_vals = quant_vals('power_mean_w_mean', s)
    t_vals = quant_vals('total_time_s_mean', s)
    axes[2].bar(xq + oi, [p * t / 3600 for p, t in zip(p_vals, t_vals)], wq, color=c, alpha=0.85)

for ax, ylabel, title in zip(axes,
        ['Power (W)', 'Power (W)', 'Energy (Wh)'],
        ['Mean GPU Power', 'Peak GPU Power', 'Energy per Image']):
    ax.set_xticks(xq); ax.set_xticklabels(QUANT_TICK, rotation=15, ha='right')
    ax.set_ylabel(ylabel); ax.set_title(title, fontweight='bold', pad=8)

handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.85) for c in Q_COLORS]
fig.legend(handles, Q_LABELS_SHORT, loc='lower center', ncol=4,
           fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.08))
fig.suptitle('Power and Energy vs Quantization', fontweight='bold', y=1.02)
fig.tight_layout()
# fig.savefig(OUT / 'fig_power_quant.pdf', bbox_inches='tight')
fig.savefig(OUT / 'fig_power_quant.png', bbox_inches='tight')
plt.close(fig)
print("✓ fig_power_quant")

print(f"\nAll figures written to {OUT}/")
