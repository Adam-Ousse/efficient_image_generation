from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import io


BASE_DIR = Path("../results/benchmark_latest")

SELECTED_PROMPTS = [11,6,4,7,3]  

MODELS_ORDER = [
    "FLUX2-Klein-4B-FP16",
    "FLUX2-Klein-4B-Q5_K_M",
    "FLUX2-Klein-4B-Q4_K_M",
    "FLUX2-Klein-4B-Q3_K_M",
    "FLUX2-Klein-4B-Q2_K",
]

SEED = 42
OUTPUT_PATH = BASE_DIR / "comparison_grid.png"


IMG_SIZE = 512  


prompt_dirs = [BASE_DIR / f"p{i}" for i in SELECTED_PROMPTS]
prompt_dirs = [p for p in prompt_dirs if p.exists()]

n_rows = len(prompt_dirs)
n_cols = len(MODELS_ORDER)


fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))

if n_rows == 1:
    axes = [axes]
if n_cols == 1:
    axes = [[ax] for ax in axes]


for row_idx, prompt_dir in enumerate(prompt_dirs):
    seed_dir = prompt_dir / f"seed_{SEED}"

    for col_idx, model_name in enumerate(MODELS_ORDER):
        ax = axes[row_idx][col_idx]
        img_path = seed_dir / f"{model_name}.png"

        if img_path.exists():
            img = Image.open(img_path)
            img.thumbnail((IMG_SIZE, IMG_SIZE))
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "Missing", ha="center", va="center")

        ax.axis("off")

        # Pretty column titles
        if row_idx == 0:
            pretty = model_name.replace("FLUX2-Klein-", "").replace("-", "\n")
            ax.set_title(pretty, fontsize=28, fontweight='bold')


buf = io.BytesIO()
plt.tight_layout()
plt.savefig(buf, format="png", dpi=150)
plt.close()

buf.seek(0)
img = Image.open(buf)

img = img.convert("P", palette=Image.ADAPTIVE)

img.save(OUTPUT_PATH, format="PNG", optimize=True, compress_level=9)

print(f"Saved compressed figure to: {OUTPUT_PATH}")