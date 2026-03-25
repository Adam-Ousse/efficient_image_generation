import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re


base_dir = "results/benchmark_flux_qwen_latest"
ocr_file = f"{base_dir}/ocr_summary.csv"
metrics_file = f"{base_dir}/comprehensive_metrics_summary.csv"
output_ocr = f"{base_dir}/ocr_report.png"
output_quality = f"{base_dir}/quality_report.png"


def parse_model_name(model: str):
    """
    Parse model name into flux and qwen parts.
    """
    if "__" in model:
        parts = model.split("__", 1)
        return (parts[0].strip(), parts[1].strip())

    return (model, "unknown")


def pivot_for_heatmap(df, value_col, flux_order=None, qwen_order=None):
    """
    Build pivot table with qwen rows and flux columns.
    """
    df = df.copy()
    df[["flux_quant", "qwen_quant"]] = pd.DataFrame(
        df["model"].apply(parse_model_name).tolist(), index=df.index
    )
    df = df.groupby(["flux_quant", "qwen_quant"], as_index=False)[value_col].mean()
    pivot = df.pivot(index="qwen_quant", columns="flux_quant", values=value_col)
    pivot.columns.name = None
    pivot.index.name = None
    if flux_order:
        pivot = pivot.reindex(columns=[c for c in flux_order if c in pivot.columns])
    if qwen_order:
        pivot = pivot.reindex(index=[r for r in qwen_order if r in pivot.index])
    return pivot


def sorted_quant_levels(levels):
    """Sort quantization labels from lowest precision to full precision."""
    def key(s):
        upper = s.upper()
        if re.search(r"\b(BF|FP)\d+\b", upper):
            bits = int(re.search(r"\d+", re.search(r"(BF|FP)\d+", upper).group()).group())
            return (1, bits)
        m = re.search(r"Q(\d+)", upper)
        bits = int(m.group(1)) if m else 99
        return (0, bits)

    return sorted(levels, key=key)



ocr_data = pd.read_csv(ocr_file)
metrics_data = pd.read_csv(metrics_file)

for df in (ocr_data, metrics_data):
    parsed = df["model"].apply(parse_model_name)
    df[["flux_quant", "qwen_quant"]] = pd.DataFrame(parsed.tolist(), index=df.index)

all_flux = sorted_quant_levels(
    list(set(ocr_data["flux_quant"]) | set(metrics_data["flux_quant"]))
)
all_qwen = sorted_quant_levels(
    list(set(ocr_data["qwen_quant"]) | set(metrics_data["qwen_quant"]))
)


plt.rcParams.update({
    "font.weight":        "bold",
    "axes.titleweight":   "bold",
    "axes.labelweight":   "bold",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "grid.color":         "#CCCCCC",
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

OCR_PALETTE = [
    "#C0392B",
    "#E8724A",   # terracotta
    "#FF6B35",
    "#2C5F8A",   # steel blue
    "#1A6B3C",
]
while len(OCR_PALETTE) < len(all_qwen):
    OCR_PALETTE.append("#555555")
OCR_PALETTE = OCR_PALETTE[:len(all_qwen)]

short_flux_ocr = [f.rsplit("-", 1)[-1] for f in all_flux]

fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
fig1.suptitle("OCR Quality vs. Quantisation", fontsize=14, fontweight="bold")

OCR_METRICS = [
    ("cer_normalized_mean", "cer_normalized_std", "CER Normalised (↓)", "CER Normalised"),
    ("wer_mean",            "wer_std",            "WER (↓)",            "WER"),
]

for ax, (metric_mean, metric_std, title, ylabel) in zip(axes1, OCR_METRICS):
    for i, qwen in enumerate(all_qwen):
        subset = ocr_data[ocr_data["qwen_quant"] == qwen].copy()
        subset = subset[subset["flux_quant"].isin(all_flux)]
        subset["flux_quant"] = pd.Categorical(subset["flux_quant"], categories=all_flux, ordered=True)
        subset = subset.sort_values("flux_quant")

        x_pos = [all_flux.index(f) for f in subset["flux_quant"]]
        y     = subset[metric_mean].values
        yerr  = subset[metric_std].values
        color = OCR_PALETTE[i]
        short_label = qwen.rsplit("-", 1)[-1]

        ax.plot(x_pos, y,
                marker="o", markersize=7, linewidth=2.2,
                color=color, label=short_label, zorder=3)
        ax.fill_between(x_pos, y - yerr, y + yerr,
                         alpha=0.12, color=color, zorder=2)

    ax.set_xticks(range(len(all_flux)))
    ax.set_xticklabels(short_flux_ocr, rotation=35, ha="right",
                        fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Flux quant", fontsize=10, fontweight="bold", labelpad=5)
    ax.set_ylabel(ylabel,       fontsize=10, fontweight="bold", labelpad=5)
    ax.yaxis.set_tick_params(labelsize=9)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")

    leg = ax.legend(title="Qwen quant", fontsize=8, title_fontsize=9,
                    loc="upper left", framealpha=0.9, edgecolor="#CCCCCC")
    leg.get_title().set_fontweight("bold")

fig1.tight_layout()
output_ocr_pdf = output_ocr.replace(".png", ".pdf")
fig1.savefig(output_ocr_pdf, bbox_inches="tight")
fig1.savefig(output_ocr,     dpi=300, bbox_inches="tight")
print(f"OCR report saved → '{output_ocr_pdf}' and '{output_ocr}'")

from matplotlib.colors import LinearSegmentedColormap

def make_cmap(light, dark, name):
    return LinearSegmentedColormap.from_list(name, [light, dark])

CMAPS = {
    "ssim":  make_cmap("#F7EDE8", "#C0392B", "ssim_cmap"),
    "fid":   make_cmap("#EAF0FB", "#2C5F8A", "fid_cmap"),
    "clip":  make_cmap("#EDF7F0", "#1A6B3C", "clip_cmap"),
}

BEST_COLOR  = "#FF6B35"
BEST_TXT    = "#FFFFFF"
NORM_TXT    = "#1A1A1A"
GRID_COLOR  = "#FFFFFF"

METRIC_CFG = [
    ("ssim_mean", "ssim_std", "SSIM (↑)",  True,  CMAPS["ssim"]),
    ("fid_score", None,       "FID (↓)",   False, CMAPS["fid"]),
    ("clip_mean", "clip_std", "CLIP (↑)",  True,  CMAPS["clip"]),
]


def build_annot(df, mean_col, std_col, flux_order, qwen_order):
    pivot_mean = pivot_for_heatmap(df, mean_col, flux_order, qwen_order)
    pivot_std  = None
    if std_col and std_col in df.columns:
        pivot_std = pivot_for_heatmap(df, std_col, flux_order, qwen_order)
    annot = np.empty(pivot_mean.shape, dtype=object)
    for i in range(pivot_mean.shape[0]):
        for j in range(pivot_mean.shape[1]):
            m = pivot_mean.iloc[i, j]
            if pd.isna(m):
                annot[i, j] = "—"
            elif pivot_std is not None and not pd.isna(pivot_std.iloc[i, j]):
                annot[i, j] = f"{m:.3f}\n±{pivot_std.iloc[i, j]:.3f}"
            else:
                annot[i, j] = f"{m:.3f}"
    return pivot_mean, annot


plt.rcParams.update({
    "font.weight":       "bold",
    "axes.titleweight":  "bold",
    "axes.labelweight":  "bold",
    "axes.grid":         False,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

short_flux = [f.rsplit("-", 1)[-1] for f in all_flux]
short_qwen = [q.rsplit("-", 1)[-1] for q in all_qwen]

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5.5))
fig2.suptitle("Image Quality vs. Quantisation", fontsize=14, fontweight="bold", y=1.01)

for ax, (mean_col, std_col, label, higher_better, cmap) in zip(axes2, METRIC_CFG):
    pivot_mean, annot_matrix = build_annot(
        metrics_data, mean_col, std_col, all_flux, all_qwen
    )
    display = pivot_mean.copy()
    display.columns = short_flux
    display.index   = short_qwen

    flat = pivot_mean.values.astype(float)
    best = None
    if not np.all(np.isnan(flat)):
        best = np.unravel_index(
            np.nanargmax(flat) if higher_better else np.nanargmin(flat),
            flat.shape,
        )
        masked = display.copy().astype(float)
        masked.iloc[best[0], best[1]] = np.nan

    sns.heatmap(
        masked if best is not None else display,
        ax=ax,
        cmap=cmap,
        annot=annot_matrix, fmt="",
        linewidths=1.2, linecolor=GRID_COLOR,
        square=True,
        cbar_kws={"shrink": 0.78, "pad": 0.02},
        annot_kws={"size": 9, "fontweight": "bold", "color": NORM_TXT},
    )

    if best is not None:
        ax.add_patch(plt.Rectangle(
            (best[1], best[0]), 1, 1,
            facecolor=BEST_COLOR, edgecolor=BEST_COLOR,
            lw=0, zorder=3,
        ))
        ax.text(
            best[1] + 0.5, best[0] + 0.5,
            annot_matrix[best[0], best[1]],
            ha="center", va="center",
            fontsize=9, fontweight="bold", color=BEST_TXT,
            zorder=4, linespacing=1.4,
        )

    ax.set_title(label, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Flux quant",  fontsize=9,  fontweight="bold", labelpad=5)
    ax.set_ylabel("Qwen quant",  fontsize=9,  fontweight="bold", labelpad=5)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

fig2.tight_layout()
output_quality_pdf = output_quality.replace(".png", ".pdf")
fig2.savefig(output_quality_pdf, bbox_inches="tight")
fig2.savefig(output_quality,     dpi=300, bbox_inches="tight")
print(f"Quality report saved → '{output_quality_pdf}' and '{output_quality}'")