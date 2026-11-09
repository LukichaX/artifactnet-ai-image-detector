import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from PIL import Image

from src.model import ArtifactNetV5


def main() -> None:
    print("=" * 68)
    print("1. GPU, CUDA, Disk & ArtifactNet v5 Verification")
    print("=" * 68)
    print(f"Python version   : {sys.version.split()[0]}")
    print(f"PyTorch version  : {torch.__version__}")
    print(f"CUDA available   : {torch.cuda.is_available()}")
    free_c_gb = shutil.disk_usage("C:\\").free / (1024 ** 3)
    print(f"Free Space on C: : {free_c_gb:.2f} GB")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        cap = torch.cuda.get_device_capability(0)
        print(f"GPU Device       : {gpu_name}")
        print(f"Physical VRAM    : {vram_gb:.2f} GB")
        print(f"Compute Cap      : sm_{cap[0]}{cap[1]}")
    else:
        print("WARNING: CUDA is not available. Training on CPU will be very slow.")

    model = ArtifactNetV5()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ArtifactNet v5   : {n_params:,} trainable parameters (100% From Scratch)")

    print("\n" + "=" * 68)
    print("2. Multi-Generator & Original Dataset Split Verification")
    print("=" * 68)

    data_dir = PROJECT_ROOT / "data"
    splits_v5 = data_dir / "splits_v5"
    if (splits_v5 / "train_split.csv").exists():
        tr = pd.read_csv(splits_v5 / "train_split.csv")
        vid = pd.read_csv(splits_v5 / "val_id_split.csv")
        vod = pd.read_csv(splits_v5 / "val_ood_split.csv")
        tod = pd.read_csv(splits_v5 / "test_ood_split.csv")
        print(f"v5 Train Split   : {len(tr):,} rows (Real={(tr['label']==0).sum():,}, AI={(tr['label']==1).sum():,})")
        print(f"v5 Val-ID Split  : {len(vid):,} rows")
        print(f"v5 Val-OOD Split : {len(vod):,} rows (Unseen Generators & Reddit)")
        print(f"v5 Test-OOD Split: {len(tod):,} rows (Unseen Generators & Reddit)")
        sample_df = tr
    else:
        train_csv = data_dir / "train.csv"
        if not train_csv.exists():
            print(f"ERROR: Could not find {train_csv}")
            return
        sample_df = pd.read_csv(train_csv)
        print(f"Loaded train.csv : {len(sample_df):,} rows (run `python src/harvest_dataset.py` to build splits_v5)")

    sample_paths = sample_df["file_name"].sample(min(200, len(sample_df)), random_state=42).tolist()
    missing = 0
    sizes = []
    for rel_path in sample_paths:
        img_path = data_dir / rel_path
        if not img_path.exists():
            missing += 1
        else:
            with Image.open(img_path) as img:
                sizes.append(img.size)

    if missing == 0 and sizes:
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        print(f"Sample integrity : {len(sizes)}/{len(sizes)} sampled images verified on disk")
        print(f"Width range      : min={min(widths)}px, max={max(widths)}px")
        print(f"Height range     : min={min(heights)}px, max={max(heights)}px")
    else:
        print(f"ERROR: {missing}/{len(sample_paths)} sampled images missing under {data_dir}")


if __name__ == "__main__":
    main()
