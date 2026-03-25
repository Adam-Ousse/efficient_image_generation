from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

BASE_DIR = Path("../results/benchmark_flux_qwen_latest")

PROMPT = "p11"
SEED = 42

FLUX_ORDER = [
    "FLUX-4B-FP16",
    "FLUX-4B-Q5_K_M",
    "FLUX-4B-Q4_K_M",
    "FLUX-4B-Q3_K_M",
    "FLUX-4B-Q2_K",
]

TEXT_ORDER = [
    "Qwen-4B-BF16",
    "Qwen-4B-Q5_K_M",
    "Qwen-4B-Q4_K_M",
    "Qwen-4B-Q3_K_M",
    "Qwen-4B-Q2_K",
]

IMG_SIZE = 512
OUTPUT_PATH = BASE_DIR / "grid_flux_vs_text_rows_fixed.pdf"
OUTPUT_PATH_jpg = BASE_DIR / "grid_flux_vs_text_rows_fixed.jpg"

def extract_flux_quant(name: str):
    return name.replace("FLUX-4B-", "")

def extract_text_quant(name: str):
    return name.replace("Qwen-4B-", "")

n_rows = len(TEXT_ORDER)
n_cols = len(FLUX_ORDER)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
prompt_dir = BASE_DIR / PROMPT / f"seed_{SEED}"

for row_idx, text_name in enumerate(TEXT_ORDER):
    for col_idx, flux_name in enumerate(FLUX_ORDER):
        ax = axes[row_idx, col_idx]

        combo_name = f"{flux_name}__{text_name}"
        img_path = prompt_dir / f"{combo_name}.png"

        if img_path.exists():
            img = Image.open(img_path)
            img.thumbnail((IMG_SIZE, IMG_SIZE))
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "Missing", ha="center", va="center", fontsize=14)

        ax.axis("off")

        if row_idx == 0:
            ax.set_title(extract_flux_quant(flux_name), fontsize=16, fontweight="bold", pad=15)

for row_idx, text_name in enumerate(TEXT_ORDER):
    axes[row_idx, 0].text(
        -0.15, 0.5,
        extract_text_quant(text_name),
        fontsize=16, fontweight="bold",
        ha="right", va="center",
        rotation=0,
        transform=axes[row_idx, 0].transAxes
    )

fig.text(0.5, 0.05 , "FLUX Quantization", ha="center", fontsize=20, fontweight="bold")
fig.text(0.05, 0.5, "Text Encoder Quantization", va="center", rotation=90, fontsize=20, fontweight="bold")

plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.08, wspace=0.05, hspace=0.05)

plt.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight")
plt.savefig(OUTPUT_PATH_jpg, format="jpg", bbox_inches="tight")
plt.close()

print(f"Saved to: {OUTPUT_PATH}")