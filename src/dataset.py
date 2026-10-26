from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


def prepare_splits(
    data_dir: Path,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    ყოფს train.csv-ს წყვილების (AI + Real) მიხედვით Train, Val და Test ნაწილებად,
    რათა თავიდან ავიცილოთ Data Leakage.
    """
    splits_dir = data_dir / "splits"
    train_path = splits_dir / "train_split.csv"
    val_path = splits_dir / "val_split.csv"
    test_path = splits_dir / "test_split.csv"

    # თუ უკვე გაყოფილია, პირდაპირ ვკითხულობთ მზა ფაილებს
    if train_path.exists() and val_path.exists() and test_path.exists():
        return (
            pd.read_csv(train_path),
            pd.read_csv(val_path),
            pd.read_csv(test_path)
        )

    splits_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_dir / "train.csv")

    # თითოეულ წყვილს (0-1, 2-3, 4-5...) ვანიჭებთ საერთო pair_id-ს
    df["pair_id"] = np.arange(len(df)) // 2
    unique_pairs = df["pair_id"].unique()

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_pairs)

    n_total = len(unique_pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_pairs = set(unique_pairs[:n_train])
    val_pairs = set(unique_pairs[n_train:n_train + n_val])
    test_pairs = set(unique_pairs[n_train + n_val:])

    train_df = df[df["pair_id"].isin(train_pairs)].drop(columns=["Unnamed: 0"], errors="ignore").reset_index(drop=True)
    val_df = df[df["pair_id"].isin(val_pairs)].drop(columns=["Unnamed: 0"], errors="ignore").reset_index(drop=True)
    test_df = df[df["pair_id"].isin(test_pairs)].drop(columns=["Unnamed: 0"], errors="ignore").reset_index(drop=True)

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_df, val_df, test_df


def get_transforms(img_size: int = 256) -> Tuple[A.Compose, A.Compose]:
    """აბრუნებს Albumentations ტრანსფორმაციებს სწავლებისა და ვალიდაციისთვის."""
    train_transform = A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.6, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ImageCompression(quality_range=(70, 100), p=0.4),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return train_transform, val_transform


class AIvsRealDataset(Dataset):
    """PyTorch Dataset სურათების სწრაფი წაკითხვისა და ტრანსფორმაციისთვის."""
    def __init__(self, df: pd.DataFrame, data_dir: Path, transform: A.Compose = None):
        self.data_dir = data_dir
        self.file_names = df["file_name"].values
        self.labels = df["label"].values.astype(np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = str(self.data_dir / self.file_names[idx])

        # OpenCV კითხულობს BGR ფორმატში, გადაგვყავს RGB-ში
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"სურათი ვერ წაიკითხა: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


def create_dataloaders(
    data_dir: Path,
    img_size: int = 256,
    batch_size: int = 128,
    num_workers: int = 8
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """ქმნის ოპტიმიზებულ DataLoader-ებს i5-12600KF-ისა და RTX 5060 Ti-სთვის."""
    train_df, val_df, test_df = prepare_splits(data_dir)
    train_tf, val_tf = get_transforms(img_size=img_size)

    train_ds = AIvsRealDataset(train_df, data_dir, transform=train_tf)
    val_ds = AIvsRealDataset(val_df, data_dir, transform=val_tf)
    test_ds = AIvsRealDataset(test_df, data_dir, transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, test_loader