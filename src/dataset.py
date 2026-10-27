import random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A


def prepare_splits(
    data_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_split_path = splits_dir / "train_split.csv"
    val_split_path = splits_dir / "val_split.csv"
    test_split_path = splits_dir / "test_split.csv"

    if train_split_path.exists() and val_split_path.exists() and test_split_path.exists():
        return (
            pd.read_csv(train_split_path),
            pd.read_csv(val_split_path),
            pd.read_csv(test_split_path),
        )

    csv_path = data_dir / "train.csv"
    df = pd.read_csv(csv_path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df["pair_id"] = df.index // 2
    unique_pairs = df["pair_id"].unique()

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_pairs)

    n_total = len(unique_pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_pairs = set(unique_pairs[:n_train])
    val_pairs = set(unique_pairs[n_train : n_train + n_val])
    test_pairs = set(unique_pairs[n_train + n_val :])

    train_df = df[df["pair_id"].isin(train_pairs)].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = df[df["pair_id"].isin(val_pairs)].reset_index(drop=True)
    test_df = df[df["pair_id"].isin(test_pairs)].reset_index(drop=True)

    train_df.to_csv(train_split_path, index=False)
    val_df.to_csv(val_split_path, index=False)
    test_df.to_csv(test_split_path, index=False)

    return train_df, val_df, test_df


def ensure_min_size_reflect(img: np.ndarray, min_size: int = 256) -> np.ndarray:
    h, w = img.shape[:2]
    pad_h = max(0, min_size - h)
    pad_w = max(0, min_size - w)
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    return img


def smash_and_reconstruct_rich_poor(
    img: np.ndarray, out_size: int = 256, patch_size: int = 32, is_train: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    img = ensure_min_size_reflect(img, min_size=out_size)
    h, w, c = img.shape
    grid_n = out_size // patch_size
    needed = grid_n * grid_n

    nh, nw = h // patch_size, w // patch_size
    cropped = img[: nh * patch_size, : nw * patch_size]
    patches = cropped.reshape(nh, patch_size, nw, patch_size, c).swapaxes(1, 2).reshape(-1, patch_size, patch_size, c)

    p_f = patches.astype(np.int16)
    dh = np.abs(p_f[:, :, 1:, :] - p_f[:, :, :-1, :]).mean(axis=(1, 2, 3))
    dv = np.abs(p_f[:, 1:, :, :] - p_f[:, :-1, :, :]).mean(axis=(1, 2, 3))
    dd1 = np.abs(p_f[:, 1:, 1:, :] - p_f[:, :-1, :-1, :]).mean(axis=(1, 2, 3))
    dd2 = np.abs(p_f[:, 1:, :-1, :] - p_f[:, :-1, 1:, :]).mean(axis=(1, 2, 3))
    l_div = dh + dv + dd1 + dd2

    order = np.argsort(l_div)[::-1]
    n_total = len(order)
    pool_k = max(8, n_total // 3)

    rich_pool = order[:pool_k]
    poor_pool = order[-pool_k:]

    if is_train:
        rich_chosen = np.random.choice(rich_pool, size=needed, replace=True)
        poor_chosen = np.random.choice(poor_pool, size=needed, replace=True)
    else:
        rich_chosen = np.resize(rich_pool, needed)
        poor_chosen = np.resize(poor_pool, needed)

    x_rich = (
        patches[rich_chosen]
        .reshape(grid_n, grid_n, patch_size, patch_size, c)
        .swapaxes(1, 2)
        .reshape(out_size, out_size, c)
    )
    x_poor = (
        patches[poor_chosen]
        .reshape(grid_n, grid_n, patch_size, patch_size, c)
        .swapaxes(1, 2)
        .reshape(out_size, out_size, c)
    )
    return x_rich, x_poor


def synthesize_pseudo_fake(img: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
    mode = random.choice(["upsample_decoder", "diffusion_smooth", "chroma_decouple", "hybrid"])

    out = img.copy()
    if mode in ("upsample_decoder", "hybrid"):
        scale = random.uniform(0.45, 0.75)
        interp_down = random.choice([cv2.INTER_AREA, cv2.INTER_LINEAR])
        interp_up = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST])
        small = cv2.resize(out, (max(32, int(w * scale)), max(32, int(h * scale))), interpolation=interp_down)
        out = cv2.resize(small, (w, h), interpolation=interp_up)

    if mode in ("diffusion_smooth", "hybrid"):
        out = cv2.bilateralFilter(out, d=5, sigmaColor=random.uniform(12, 28), sigmaSpace=random.uniform(12, 28))

    if mode in ("chroma_decouple", "hybrid"):
        ycrcb = cv2.cvtColor(out, cv2.COLOR_RGB2YCrCb)
        cr_small = cv2.resize(ycrcb[:, :, 1], (max(16, w // 2), max(16, h // 2)), interpolation=cv2.INTER_LINEAR)
        cb_small = cv2.resize(ycrcb[:, :, 2], (max(16, w // 2), max(16, h // 2)), interpolation=cv2.INTER_LINEAR)
        ycrcb[:, :, 1] = cv2.resize(cr_small, (w, h), interpolation=cv2.INTER_LINEAR)
        ycrcb[:, :, 2] = cv2.resize(cb_small, (w, h), interpolation=cv2.INTER_LINEAR)
        out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    return out


def apply_realistic_compression(img_rgb: np.ndarray, p: float = 0.65) -> np.ndarray:
    if random.random() > p:
        return img_rgb

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    if random.random() < 0.75:
        quality = random.randint(45, 98)
        ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    else:
        quality = random.randint(50, 98)
        ok, enc = cv2.imencode(".webp", img_bgr, [int(cv2.IMWRITE_WEBP_QUALITY), quality])

    if not ok:
        return img_rgb
    dec_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if dec_bgr is None:
        return img_rgb
    return cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB)


def apply_safe_patch_masking_np(img: np.ndarray, patch_size: int = 32, max_ratio: float = 0.25) -> np.ndarray:
    h, w, _ = img.shape
    ratio = random.uniform(0.05, max_ratio)
    nh, nw = h // patch_size, w // patch_size
    total_patches = nh * nw
    num_mask = int(total_patches * ratio)
    if num_mask <= 0:
        return img

    out = img.copy()
    mask_indices = random.sample(range(total_patches), num_mask)
    for idx in mask_indices:
        r, col = divmod(idx, nw)
        out[r * patch_size : (r + 1) * patch_size, col * patch_size : (col + 1) * patch_size, :] = 0
    return out


def get_train_augmentations() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.5),
        ]
    )


class ForensicPairDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        img_size: int = 256,
        is_train: bool = True,
        pseudo_fake_prob: float = 0.20,
    ) -> None:
        self.file_names = df["file_name"].astype(str).tolist()
        self.labels = df["label"].astype(np.float32).tolist()
        self.data_dir = data_dir
        self.img_size = img_size
        self.is_train = is_train
        self.pseudo_fake_prob = pseudo_fake_prob
        self.aug = get_train_augmentations() if is_train else None

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path = self.data_dir / self.file_names[idx]
        label = self.labels[idx]

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            img_rgb = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self.is_train and label == 0.0 and random.random() < self.pseudo_fake_prob:
            img_rgb = synthesize_pseudo_fake(img_rgb)
            label = 1.0

        if self.is_train:
            img_rgb = apply_realistic_compression(img_rgb, p=0.65)
            if self.aug is not None:
                img_rgb = self.aug(image=img_rgb)["image"]

        x_rich_np, x_poor_np = smash_and_reconstruct_rich_poor(
            img_rgb, out_size=self.img_size, patch_size=32, is_train=self.is_train
        )

        if self.is_train and random.random() < 0.5:
            x_rich_np = apply_safe_patch_masking_np(x_rich_np, patch_size=32, max_ratio=0.25)

        # ვაბრუნებთ მსუბუქ uint8 ტენზორებს (RAM-ს ზოგავს 4-ჯერ; GPU გადაიყვანს float-ში)
        x_rich = torch.from_numpy(np.ascontiguousarray(x_rich_np.transpose(2, 0, 1)))
        x_poor = torch.from_numpy(np.ascontiguousarray(x_poor_np.transpose(2, 0, 1)))

        return x_rich, x_poor, torch.tensor(label, dtype=torch.float32)


def create_dataloaders(
    data_dir: Path,
    img_size: int = 256,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_df, val_df, test_df = prepare_splits(data_dir)

    train_ds = ForensicPairDataset(train_df, data_dir, img_size=img_size, is_train=True, pseudo_fake_prob=0.20)
    val_ds = ForensicPairDataset(val_df, data_dir, img_size=img_size, is_train=False, pseudo_fake_prob=0.0)
    test_ds = ForensicPairDataset(test_df, data_dir, img_size=img_size, is_train=False, pseudo_fake_prob=0.0)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return train_loader, val_loader, test_loader