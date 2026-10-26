import torch
import pandas as pd
from pathlib import Path
from PIL import Image
import random

print("=" * 50)
print("1. ვიდეოკარტის და PyTorch CUDA-ს შემოწმება")
print("=" * 50)
print(f"PyTorch ვერსია: {torch.__version__}")
print(f"CUDA ხელმისაწვდომია: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    capability = torch.cuda.get_device_capability(0)
    print(f"ვიდეოკარტა: {gpu_name}")
    print(f"VRAM მოცულობა: {vram_gb:.2f} GB")
    print(f"Compute Capability: sm_{capability[0]}{capability[1]}0")

    # რეალური ტესტი ვიდეოკარტის ბირთვებზე (BFloat16)
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    y = torch.matmul(x, x)
    print("BFloat16 GPU ტესტი: წარმატებით შესრულდა! 🚀")
else:
    print("❌ ყურადღება: PyTorch ვერ ხედავს ვიდეოკარტას!")

print("\n" + "=" * 50)
print("2. დატასეტის და რეზოლუციების შემოწმება")
print("=" * 50)

# ვპოულობთ data ფოლდერს
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_CSV = DATA_DIR / "train.csv"

if not TRAIN_CSV.exists():
    print(f"❌ ვერ ვიპოვე {TRAIN_CSV}! შეამოწმე ფოლდერების სტრუქტურა.")
else:
    df = pd.read_csv(TRAIN_CSV)
    print(f"train.csv წარმატებით ჩაიტვირთა! სულ ჩანაწერი: {len(df)}")

    # შევამოწმოთ შემთხვევითი 200 სურათი დისკზე
    sample_paths = df["file_name"].sample(200, random_state=42).tolist()
    missing = 0
    sizes = []

    for rel_path in sample_paths:
        img_path = DATA_DIR / rel_path
        if not img_path.exists():
            missing += 1
        else:
            with Image.open(img_path) as img:
                sizes.append(img.size)

    if missing == 0:
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        print(f"✅ 200-ვე სატესტო სურათი ადგილზეა და იხსნება!")
        print(f"   • სიგანე (Width): მინ={min(widths)}px, მაქს={max(widths)}px")
        print(f"   • სიმაღლე (Height): მინ={min(heights)}px, მაქს={max(heights)}px")
        print(f"   • მაგალითი რეზოლუციები: {sizes[:5]}")
    else:
        print(f"❌ 200-დან {missing} სურათი ვერ მოიძებნა მისამართზე: {DATA_DIR / sample_paths[0]}")