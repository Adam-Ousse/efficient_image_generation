from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

BASE_DIR = Path("../results/benchmark_latest")

PROMPTS = ["p1", "p4", "p6", "p8", "p9", "p10", "p11", "p13"]
SEED = 42

MODELS = [
    "FLUX2-Klein-dev-FP16",
    "FLUX2-Klein-9B-FP16",
    "FLUX2-Klein-4B-FP16",
]

IMG_SIZE = 768

def format_name(name: str):
    return name.replace("FLUX2-Klein-", "")


half = len(PROMPTS) // 2
left_prompts = PROMPTS[:half]
right_prompts = PROMPTS[half:]

n_rows = half
n_cols = 6

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

for row_idx in range(n_rows):

    prompt = left_prompts[row_idx]
    prompt_dir = BASE_DIR / prompt / f"seed_{SEED}"

    for col_idx, model_name in enumerate(MODELS):
        ax = axes[row_idx][col_idx]

        img_path = prompt_dir / f"{model_name}.png"

        if img_path.exists():
            img = Image.open(img_path)
            img.thumbnail((IMG_SIZE, IMG_SIZE))
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "Missing", ha="center", va="center", fontsize=12)

        ax.axis("off")

        if row_idx == 0:
            ax.set_title(format_name(model_name), fontsize=14, fontweight="bold")

    prompt = right_prompts[row_idx]
    prompt_dir = BASE_DIR / prompt / f"seed_{SEED}"

    for col_idx, model_name in enumerate(MODELS):
        ax = axes[row_idx][col_idx + 3]

        img_path = prompt_dir / f"{model_name}.png"

        if img_path.exists():
            img = Image.open(img_path)
            img.thumbnail((IMG_SIZE, IMG_SIZE))
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "Missing", ha="center", va="center", fontsize=12)

        ax.axis("off")

        if row_idx == 0:
            ax.set_title(format_name(model_name), fontsize=14, fontweight="bold")

fig.text(0.1, 0.5, "Prompt", va="center", rotation=90, fontsize=18, fontweight="bold")

plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.08, wspace=0.05, hspace=0.05)

OUTPUT_PDF = BASE_DIR / "grid_distill_6cols.pdf"
OUTPUT_JPG = BASE_DIR / "grid_distill_6cols.jpg"

plt.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
plt.savefig(OUTPUT_JPG, format="jpg", bbox_inches="tight")
plt.close()

print(f"Saved to: {OUTPUT_PDF}")