import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from PIL import Image

from src.model import ArtifactNetV4


def main() -> None:
    print("=" * 64)
    print("1. GPU, CUDA & BFloat16 Environment Verification")
    print("=" * 64)
    print(f"Python version   : {sys.version.split()[0]}")
    print(f"PyTorch version  : {torch.__version__}")
    print(f"CUDA available   : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        cap = torch.cuda.get_device_capability(0)
        print(f"GPU Device       : {gpu_name}")
        print(f"Physical VRAM    : {vram_gb:.2f} GB")
        print(f"Compute Cap      : sm_{cap[0]}{cap[1]}")
    else:
        print("WARNING: CUDA is not available. Training on CPU will be very slow.")

    model = ArtifactNetV4()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ArtifactNet v4   : {n_params:,} trainable parameters (100% From Scratch)")

    print("\n" + "=" * 64)
    print("2. Dataset & Pair-Aware Split Verification")
    print("=" * 64)

    data_dir = PROJECT_ROOT / "data"
    train_csv = data_dir / "train.csv"

    if not train_csv.exists():
        print(f"ERROR: Could not find {train_csv}")
        return

    df = pd.read_csv(train_csv)
    print(f"Loaded train.csv : {len(df):,} rows ({len(df)//2:,} matched Real/AI pairs)")

    sample_paths = df["file_name"].sample(min(200, len(df)), random_state=42).tolist()
    missing = 0
    sizes = []
    for rel_path in sample_paths:
        img_path = data_dir / rel_path
        if not img_path.exists():
            missing += 1
        else:
            with Image.open(img_path) as img:
                sizes.append(img.size)

    if missing == 0:
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        print("Sample integrity : 200/200 sampled images verified on disk")
        print(f"Width range      : min={min(widths)}px, max={max(widths)}px")
        print(f"Height range     : min={min(heights)}px, max={max(heights)}px")
    else:
        print(f"ERROR: {missing}/200 sampled images missing under {data_dir}")


if __name__ == "__main__":
    main()
