from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


CANONICAL_MAX_LONG_SIDE = 1024


def ensure_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """Normalizes any HxW (grayscale), HxWx4 (RGBA), or float/uint16 image into contiguous HxWx3 uint8 RGB."""
    if img is None:
        return np.zeros((256, 256, 3), dtype=np.uint8)
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating):
            scale = 255.0 if float(np.nanmax(img)) <= 1.5 else 1.0
            img = np.clip(img * scale, 0.0, 255.0).astype(np.uint8)
        elif img.dtype == np.uint16:
            img = (img >> 8).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        img = np.clip(rgb * alpha + 128.0 * (1.0 - alpha), 0.0, 255.0).astype(np.uint8)
    elif img.ndim == 3 and img.shape[2] > 4:
        img = img[:, :, :3]
    return np.ascontiguousarray(img)


def strip_borders_and_canvas(img_rgb: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """
    Detects monochrome/vintage color profiles and strips solid black/white letterbox or screenshot borders
    so uniform padding never pollutes smooth-patch (`x_poor`) forensic statistics.
    """
    img_rgb = ensure_rgb_uint8(img_rgb)
    img_f = img_rgb.astype(np.float32)

    rg_diff = np.abs(img_f[:, :, 0] - img_f[:, :, 1])
    gb_diff = np.abs(img_f[:, :, 1] - img_f[:, :, 2])
    rb_diff = np.abs(img_f[:, :, 0] - img_f[:, :, 2])
    pixel_chroma = (rg_diff + gb_diff + rb_diff) / 3.0

    mean_chroma = float(np.mean(pixel_chroma))
    p98_chroma = float(np.percentile(pixel_chroma, 98))

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat_98 = float(np.percentile(hsv[:, :, 1], 98))
    is_monochrome = (mean_chroma < 4.5 and p98_chroma < 14.0) or (sat_98 < 22.0)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.abs(gx) + np.abs(gy)

    is_canvas_px = ((gray > 246.0) | (gray < 9.0)) & (grad < 4.5)
    canvas_ratio = float(np.mean(is_canvas_px))

    if canvas_ratio > 0.04:
        row_active = np.mean(~is_canvas_px, axis=1) > 0.15
        col_active = np.mean(~is_canvas_px, axis=0) > 0.15
        y_idx = np.where(row_active)[0]
        x_idx = np.where(col_active)[0]
        if len(y_idx) > 64 and len(x_idx) > 64:
            y1, y2 = int(y_idx[0]), int(y_idx[-1]) + 1
            x1, x2 = int(x_idx[0]), int(x_idx[-1]) + 1
            if (y2 - y1) >= 128 and (x2 - x1) >= 128:
                return img_rgb[y1:y2, x1:x2], canvas_ratio, is_monochrome

    return img_rgb, canvas_ratio, is_monochrome


def canonicalize_scale(
    img_rgb: np.ndarray,
    max_long_side: int = CANONICAL_MAX_LONG_SIDE,
    target_long_side: int | None = None,
) -> np.ndarray:
    """
    Harmonizes input pixel pitch so 32x32 micro-tiles represent physical scene structure consistently
    across 768px web images, 1024px-1536px generator outputs, and 4000px camera photos.
    Uses INTER_AREA (exact box integration / sensor binning) when downsampling.
    """
    h, w = img_rgb.shape[:2]
    long_side = max(h, w)
    if target_long_side is not None:
        if abs(long_side - target_long_side) <= 8:
            return img_rgb
        scale = target_long_side / max(1, long_side)
    elif long_side > max_long_side:
        scale = max_long_side / long_side
    else:
        return img_rgb

    nh = max(64, int(round(h * scale)))
    nw = max(64, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(img_rgb, (nw, nh), interpolation=interp)


def ensure_min_size_reflect(img: np.ndarray, min_size: int = 256) -> np.ndarray:
    h, w = img.shape[:2]
    pad_h = max(0, min_size - h)
    pad_w = max(0, min_size - w)
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    return img


def smash_and_reconstruct_rich_poor(
    img: np.ndarray,
    out_size: int = 256,
    patch_size: int = 32,
    is_train: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    PatchCraft-inspired Rich/Poor Tile Decomposition with Clipped-Exposure Filtering:
    - x_rich: highest 4-directional micro-texture variation tiles (edges, hair, fabric, pores).
    - x_poor: smoothest UNCLIPPED midtone tiles (bokeh, sky, walls), excluding blown 255 / crushed 0 patches
              where 8-bit saturation artificially zeroes out camera sensor noise.
    """
    img = ensure_min_size_reflect(ensure_rgb_uint8(img), min_size=out_size)
    h, w, c = img.shape
    grid_n = out_size // patch_size
    needed = grid_n * grid_n

    nh, nw = h // patch_size, w // patch_size
    cropped = img[: nh * patch_size, : nw * patch_size]
    patches = cropped.reshape(nh, patch_size, nw, patch_size, c).swapaxes(1, 2).reshape(-1, patch_size, patch_size, c)

    if patches.shape[0] < 96 and h >= patch_size * 2 + 16 and w >= patch_size * 2 + 16:
        sh, sw = (h - 16) // patch_size, (w - 16) // patch_size
        if sh >= 2 and sw >= 2:
            sub = img[16 : 16 + sh * patch_size, 16 : 16 + sw * patch_size]
            extra = sub.reshape(sh, patch_size, sw, patch_size, c).swapaxes(1, 2).reshape(-1, patch_size, patch_size, c)
            patches = np.concatenate([patches, extra], axis=0)

    p_f = patches.astype(np.float32)
    dh = np.abs(p_f[:, :, 1:, :] - p_f[:, :, :-1, :]).mean(axis=(1, 2, 3))
    dv = np.abs(p_f[:, 1:, :, :] - p_f[:, :-1, :, :]).mean(axis=(1, 2, 3))
    dd1 = np.abs(p_f[:, 1:, 1:, :] - p_f[:, :-1, :-1, :]).mean(axis=(1, 2, 3))
    dd2 = np.abs(p_f[:, 1:, :-1, :] - p_f[:, :-1, 1:, :]).mean(axis=(1, 2, 3))
    l_div = dh + dv + dd1 + dd2

    patch_mean = p_f.mean(axis=(1, 2, 3))
    clipped_frac = ((p_f >= 251.0) | (p_f <= 4.0)).mean(axis=(1, 2, 3))
    unclipped_mask = (patch_mean >= 10.0) & (patch_mean <= 245.0) & (clipped_frac < 0.35) & (l_div > 0.65)

    if int(np.sum(unclipped_mask)) >= 16:
        cand_patches = patches[unclipped_mask]
        cand_div = l_div[unclipped_mask]
    else:
        non_flat = l_div > 0.4
        if int(np.sum(non_flat)) >= 12:
            cand_patches = patches[non_flat]
            cand_div = l_div[non_flat]
        else:
            cand_patches = patches
            cand_div = l_div

    order = np.argsort(cand_div)[::-1]
    pool_k = max(6, len(order) // 3)
    rich_pool = order[:pool_k]
    poor_pool = order[-pool_k:]

    if is_train:
        gen = rng if rng is not None else np.random.default_rng()
        rich_chosen = gen.choice(rich_pool, size=needed, replace=True)
        poor_chosen = gen.choice(poor_pool, size=needed, replace=True)
    else:
        rich_chosen = np.resize(rich_pool, needed)
        poor_chosen = np.resize(poor_pool, needed)

    x_rich = (
        cand_patches[rich_chosen]
        .reshape(grid_n, grid_n, patch_size, patch_size, c)
        .swapaxes(1, 2)
        .reshape(out_size, out_size, c)
    )
    x_poor = (
        cand_patches[poor_chosen]
        .reshape(grid_n, grid_n, patch_size, patch_size, c)
        .swapaxes(1, 2)
        .reshape(out_size, out_size, c)
    )
    return np.ascontiguousarray(x_rich), np.ascontiguousarray(x_poor)


def prepare_forensic_views(
    raw_img_rgb: np.ndarray,
    img_size: int = 256,
    patch_size: int = 32,
    max_long_side: int = CANONICAL_MAX_LONG_SIDE,
    target_long_side: int | None = None,
    strip_borders: bool = True,
    is_train: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Single Canonical Preprocessing Contract shared across Training, Validation, CLI Evaluation, and Gradio UI.
    Returns (x_global, x_rich, x_poor, metadata) as uint8 HxWx3 arrays.
    """
    img = ensure_rgb_uint8(raw_img_rgb)
    orig_h, orig_w = img.shape[:2]

    if strip_borders:
        clean_img, canvas_ratio, is_monochrome = strip_borders_and_canvas(img)
    else:
        clean_img, canvas_ratio, is_monochrome = img, 0.0, False

    canon_img = canonicalize_scale(
        clean_img, max_long_side=max_long_side, target_long_side=target_long_side
    )
    x_global = cv2.resize(canon_img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    x_rich, x_poor = smash_and_reconstruct_rich_poor(
        canon_img, out_size=img_size, patch_size=patch_size, is_train=is_train, rng=rng
    )
    meta = {
        "orig_size": (orig_w, orig_h),
        "canon_size": (canon_img.shape[1], canon_img.shape[0]),
        "canvas_ratio": canvas_ratio,
        "is_monochrome": is_monochrome,
    }
    return np.ascontiguousarray(x_global), x_rich, x_poor, meta


def prepare_splits(
    data_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Legacy single-dataset split loader (`data/splits/`)."""
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
    unique_pairs = df["pair_id"].unique().copy()

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


def prepare_splits_v5(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads `data/splits_v5/{train_split,val_id_split,val_ood_split,test_ood_split}.csv`
    created by `src/harvest_dataset.py`. Falls back gracefully to `prepare_splits` if v5 splits do not exist yet.
    """
    v5_dir = data_dir / "splits_v5"
    p_train = v5_dir / "train_split.csv"
    p_val_id = v5_dir / "val_id_split.csv"
    p_val_ood = v5_dir / "val_ood_split.csv"
    p_test_ood = v5_dir / "test_ood_split.csv"

    if p_train.exists() and p_val_id.exists() and p_val_ood.exists() and p_test_ood.exists():
        return (
            pd.read_csv(p_train),
            pd.read_csv(p_val_id),
            pd.read_csv(p_val_ood),
            pd.read_csv(p_test_ood),
        )

    tr, vl, ts = prepare_splits(data_dir)
    for df_tmp in (tr, vl, ts):
        if "family" not in df_tmp.columns:
            df_tmp["family"] = np.where(df_tmp["label"].astype(int) == 0, 0, 11)
        if "generator" not in df_tmp.columns:
            df_tmp["generator"] = np.where(df_tmp["label"].astype(int) == 0, "shutterstock_real", "deepmedia_v1")
        if "source_dataset" not in df_tmp.columns:
            df_tmp["source_dataset"] = "legacy_train_v1"
    return tr, vl, vl.copy(), ts


def _encode_decode_rgb(img_rgb: np.ndarray, codec: str, quality: int) -> np.ndarray:
    if codec == "none":
        return img_rgb
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok = False
    enc = None
    if codec == "jpeg_444":
        q = int(np.clip(quality, 60, 99))
        ok, enc = cv2.imencode(
            ".jpg",
            bgr,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                q,
                int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR),
                int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444),
            ],
        )
    elif codec == "jpeg_420":
        q = int(np.clip(quality, 46, 96))
        ok, enc = cv2.imencode(
            ".jpg",
            bgr,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                q,
                int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR),
                int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420),
            ],
        )
    elif codec == "webp":
        q = int(np.clip(quality, 50, 96))
        ok, enc = cv2.imencode(".webp", bgr, [int(cv2.IMWRITE_WEBP_QUALITY), q])

    if ok and enc is not None:
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if dec is not None:
            return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    return img_rgb


def sample_internet_laundering_params(rng: np.random.Generator) -> dict:
    """
    Samples a realistic multi-hop internet / social-media / messenger laundering pipeline
    applied IDENTICALLY to both Real_k and AI_k in every matched training pair:
      - Regime 'clean' (22%): Fresh camera/generator upload (only sub-block crop shift + dihedral symmetry + clean/PNG/Q96).
      - Regime '1hop_social' (30%): Single messenger/platform delivery (WhatsApp, Instagram, Telegram, X/WebP CDN).
      - Regime '2hop_laundered' (32%): Uploaded -> Compressed/Resized -> Downloaded -> Forwarded/Saved (including compressed-then-PNG!).
      - Regime 'harsh_web' (16%): Thumbnail/screenshot/ISP blur-sharpen + heavy Q54-70 re-encoding.
    """
    interps = [cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
    r_roll = float(rng.random())
    if r_roll < 0.22:
        regime = "clean"
    elif r_roll < 0.52:
        regime = "1hop_social"
    elif r_roll < 0.84:
        regime = "2hop_laundered"
    else:
        regime = "harsh_web"

    isp_roll = float(rng.random())
    if isp_roll < 0.18:
        isp_mode = "sharpen"
    elif isp_roll < 0.30:
        isp_mode = "soften"
    elif isp_roll < 0.42:
        isp_mode = "iso_grain"
    else:
        isp_mode = "none"

    # Hop 1 & Hop 2 codec selection per regime
    if regime == "clean":
        c_roll = float(rng.random())
        hop1_codec = "none" if c_roll < 0.55 else ("jpeg_444" if c_roll < 0.85 else "webp")
        hop1_q = int(rng.integers(92, 99))
        hop1_scale = float(rng.uniform(0.92, 1.08)) if rng.random() < 0.25 else 1.0
        do_hop2 = False
        hop2_codec, hop2_q, hop2_scale = "none", 95, 1.0
    elif regime == "1hop_social":
        c_roll = float(rng.random())
        hop1_codec = "jpeg_420" if c_roll < 0.72 else ("webp" if c_roll < 0.92 else "jpeg_444")
        hop1_q = int(rng.integers(70, 86))
        hop1_scale = float(rng.uniform(0.68, 1.16))
        do_hop2 = False
        hop2_codec, hop2_q, hop2_scale = "none", 90, 1.0
    elif regime == "2hop_laundered":
        hop1_codec = "jpeg_420" if rng.random() < 0.70 else "webp"
        hop1_q = int(rng.integers(74, 89))
        hop1_scale = float(rng.uniform(0.64, 1.22))
        do_hop2 = True
        c2_roll = float(rng.random())
        # 20% of 2-hop chains end in lossless 'none' (simulates screenshotting or saving a compressed web image as PNG!)
        if c2_roll < 0.55:
            hop2_codec = "jpeg_420"
        elif c2_roll < 0.80:
            hop2_codec = "webp"
        else:
            hop2_codec = "none"
        hop2_q = int(rng.integers(65, 83))
        hop2_scale = float(rng.uniform(0.72, 1.12))
    else:
        # harsh_web
        hop1_codec = "jpeg_420" if rng.random() < 0.75 else "webp"
        hop1_q = int(rng.integers(52, 71))
        hop1_scale = float(rng.uniform(0.46, 0.78))
        do_hop2 = bool(rng.random() < 0.50)
        hop2_codec = "jpeg_420" if rng.random() < 0.70 else "none"
        hop2_q = int(rng.integers(54, 74))
        hop2_scale = float(rng.uniform(0.82, 1.20))

    return {
        "regime": regime,
        "do_grid_shift": bool(rng.random() < 0.88),
        "dy1": int(rng.integers(0, 8)),
        "dx1": int(rng.integers(0, 8)),
        "dy2": int(rng.integers(0, 8)),
        "dx2": int(rng.integers(0, 8)),
        "hflip": bool(rng.random() < 0.50),
        "vflip": bool(rng.random() < 0.15),
        "rot90_k": int(rng.integers(0, 4)) if rng.random() < 0.35 else 0,
        "hop1_scale": hop1_scale,
        "hop1_interp": int(interps[int(rng.integers(0, len(interps)))]),
        "hop1_codec": hop1_codec,
        "hop1_q": hop1_q,
        "isp_mode": isp_mode,
        "sharpen_alpha": float(rng.uniform(0.14, 0.38)),
        "blur_sigma": float(rng.uniform(0.35, 0.92)),
        "grain_sigma": float(rng.uniform(1.0, 4.0)),
        "grain_seed": int(rng.integers(0, 2**31 - 1)),
        "do_photo": bool(rng.random() < 0.36),
        "alpha_gain": float(rng.uniform(0.91, 1.09)),
        "beta_bias": float(rng.uniform(-9.0, 9.0)),
        "gamma": float(rng.uniform(0.91, 1.10)),
        "to_monochrome": bool(rng.random() < 0.04),
        "do_hop2": do_hop2,
        "hop2_shift_y": int(rng.integers(1, 6)),
        "hop2_shift_x": int(rng.integers(1, 6)),
        "hop2_scale": hop2_scale,
        "hop2_interp": int(interps[int(rng.integers(0, len(interps)))]),
        "hop2_sharpen": float(rng.uniform(0.10, 0.26)) if rng.random() < 0.35 else 0.0,
        "hop2_codec": hop2_codec,
        "hop2_q": hop2_q,
        "do_mask": bool(rng.random() < 0.25),
        "mask_count": int(rng.integers(2, 9)),
        "mask_seed": int(rng.integers(0, 2**31 - 1)),
        "patch_seed": int(rng.integers(0, 2**31 - 1)),
    }


def apply_internet_laundering(img_rgb: np.ndarray, p: dict) -> np.ndarray:
    out = ensure_rgb_uint8(img_rgb)
    h, w = out.shape[:2]

    if p["do_grid_shift"] and h > 96 and w > 96:
        y1 = min(p["dy1"], h // 8)
        x1 = min(p["dx1"], w // 8)
        y2 = h - min(p["dy2"], h // 8)
        x2 = w - min(p["dx2"], w // 8)
        if (y2 - y1) >= 64 and (x2 - x1) >= 64:
            out = out[y1:y2, x1:x2]

    if p["hflip"]:
        out = cv2.flip(out, 1)
    if p["vflip"]:
        out = cv2.flip(out, 0)
    if p["rot90_k"] > 0:
        out = np.rot90(out, k=p["rot90_k"]).copy()

    # Hop 1: Scale + ISP / Photometric + Platform Codec 1
    if abs(p["hop1_scale"] - 1.0) > 1e-3:
        ch, cw = out.shape[:2]
        nh = int(np.clip(round(ch * p["hop1_scale"]), 176, 1152))
        nw = int(np.clip(round(cw * p["hop1_scale"]), 176, 1152))
        out = cv2.resize(out, (nw, nh), interpolation=p["hop1_interp"])

    isp = p["isp_mode"]
    if isp == "sharpen":
        blur = cv2.GaussianBlur(out, (0, 0), sigmaX=1.0)
        a = p["sharpen_alpha"]
        out = cv2.addWeighted(out, 1.0 + a, blur, -a, 0)
    elif isp == "soften":
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=p["blur_sigma"])
    elif isp == "iso_grain":
        g_rng = np.random.default_rng(p["grain_seed"])
        out_f = out.astype(np.float32)
        shot_scale = np.sqrt(np.clip(out_f / 255.0, 0.05, 1.0))
        noise = g_rng.normal(0.0, p["grain_sigma"], size=out.shape).astype(np.float32) * shot_scale
        out = np.clip(out_f + noise, 0.0, 255.0).astype(np.uint8)

    if p["do_photo"]:
        out_f = out.astype(np.float32) * p["alpha_gain"] + p["beta_bias"]
        out_f = np.clip(out_f / 255.0, 0.0, 1.0) ** p["gamma"]
        out = np.clip(out_f * 255.0, 0.0, 255.0).astype(np.uint8)

    if p["to_monochrome"]:
        gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        out = np.stack([gray, gray, gray], axis=-1)

    out = _encode_decode_rgb(out, p["hop1_codec"], p["hop1_q"])

    # Hop 2: Second platform / forward / download / screenshot-save hop
    if p["do_hop2"]:
        ch, cw = out.shape[:2]
        if ch > 96 and cw > 96:
            sy = min(p["hop2_shift_y"], ch // 12)
            sx = min(p["hop2_shift_x"], cw // 12)
            out = out[sy:, sx:]
            ch, cw = out.shape[:2]

        if abs(p["hop2_scale"] - 1.0) > 1e-3:
            nh = int(np.clip(round(ch * p["hop2_scale"]), 176, 1088))
            nw = int(np.clip(round(cw * p["hop2_scale"]), 176, 1088))
            out = cv2.resize(out, (nw, nh), interpolation=p["hop2_interp"])

        if p["hop2_sharpen"] > 0.0:
            blur2 = cv2.GaussianBlur(out, (0, 0), sigmaX=0.9)
            a2 = p["hop2_sharpen"]
            out = cv2.addWeighted(out, 1.0 + a2, blur2, -a2, 0)

        out = _encode_decode_rgb(out, p["hop2_codec"], p["hop2_q"])

    return out


# Backward-compatible aliases
sample_shared_aug_params = sample_internet_laundering_params
apply_shared_aug = apply_internet_laundering


def apply_patch_mask_sync(img: np.ndarray, count: int, seed: int, patch_size: int = 32) -> np.ndarray:
    h, w, _ = img.shape
    nh, nw = h // patch_size, w // patch_size
    total = nh * nw
    k = min(count, max(0, total // 4))
    if k <= 0:
        return img
    out = img.copy()
    fill_val = out.mean(axis=(0, 1), keepdims=True).astype(np.uint8)
    m_rng = np.random.default_rng(seed)
    chosen = m_rng.choice(total, size=k, replace=False)
    for idx in chosen:
        r, col = divmod(int(idx), nw)
        out[r * patch_size : (r + 1) * patch_size, col * patch_size : (col + 1) * patch_size, :] = fill_val
    return out


def apply_eval_perturbation(img_rgb: np.ndarray, mode: str) -> np.ndarray:
    """
    Deterministic out-of-domain & internet-laundering transforms for validation & final stress-testing:
      - 'clean': untouched image
      - 'whatsapp_1hop': 1280px max-side INTER_AREA + JPEG Q74 (4:2:0) messenger delivery
      - 'social_2hop_chain': Upload (0.82x + sharpen + Q82) -> Download -> Forward/Save (sub-block shift + 0.88x + WebP Q76 -> JPEG Q73)
      - 'webp_cdn_q75': Sub-block shift + 0.90x resize + WebP Q75 CDN delivery
      - 'screenshot_reencode': Compressed web JPEG Q72 -> 1.12x screen bilinear scale + gamma 1.04 -> saved lossless (PNG-like)
      - 'phone_isp': Bilateral denoise + unsharp mask + JPEG Q90
      - 'upscale_1_8x': 1.8x bicubic upscale
      - 'downscale_0_68x_png': Sub-block shift + 0.68x INTER_AREA lossless
      - 'jpeg_q96_444': Sub-block shift + 1.15x Lanczos + JPEG Q96 (4:4:4)
      - 'fft_hf_knockout': Notches out radial high-frequency ring [0.50, 0.88] in 2D FFT
    """
    img = ensure_rgb_uint8(img_rgb)
    h, w = img.shape[:2]

    if mode == "clean":
        return img

    if mode == "whatsapp_1hop":
        long_s = max(h, w)
        if long_s > 1280 or long_s < 800:
            sc = min(1280.0 / max(1, long_s), 0.92)
            nh, nw = max(160, int(round(h * sc))), max(160, int(round(w * sc)))
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        return _encode_decode_rgb(img, "jpeg_420", 74)

    if mode == "social_2hop_chain":
        # Hop 1: Social platform upload (0.82x bicubic + subtle unsharp + JPEG Q82 4:2:0)
        nh1, nw1 = max(176, int(round(h * 0.82))), max(176, int(round(w * 0.82)))
        hop1 = cv2.resize(img, (nw1, nh1), interpolation=cv2.INTER_CUBIC)
        blur1 = cv2.GaussianBlur(hop1, (0, 0), 1.0)
        hop1 = cv2.addWeighted(hop1, 1.20, blur1, -0.20, 0)
        hop1 = _encode_decode_rgb(hop1, "jpeg_420", 82)
        # Hop 2: Downloaded, misaligned DCT grid, resized, WebP CDN + messenger JPEG Q73
        ch, cw = hop1.shape[:2]
        if ch > 96 and cw > 96:
            hop1 = hop1[3:, 5:]
            ch, cw = hop1.shape[:2]
        nh2, nw2 = max(160, int(round(ch * 0.88))), max(160, int(round(cw * 0.88)))
        hop2 = cv2.resize(hop1, (nw2, nh2), interpolation=cv2.INTER_AREA)
        hop2 = _encode_decode_rgb(hop2, "webp", 76)
        return _encode_decode_rgb(hop2, "jpeg_420", 73)

    if mode == "webp_cdn_q75":
        cropped = img[2 : max(66, h - 2), 4 : max(68, w - 2)] if (h > 96 and w > 96) else img
        ch, cw = cropped.shape[:2]
        resized = cv2.resize(
            cropped, (max(160, int(round(cw * 0.90))), max(160, int(round(ch * 0.90)))), interpolation=cv2.INTER_LINEAR
        )
        return _encode_decode_rgb(resized, "webp", 75)

    if mode == "screenshot_reencode":
        nh1, nw1 = max(160, int(round(h * 0.76))), max(160, int(round(w * 0.76)))
        web_img = _encode_decode_rgb(cv2.resize(img, (nw1, nh1), interpolation=cv2.INTER_AREA), "jpeg_420", 72)
        ch, cw = web_img.shape[:2]
        scr = cv2.resize(
            web_img, (int(round(cw * 1.12)), int(round(ch * 1.12))), interpolation=cv2.INTER_LINEAR
        )
        scr_f = np.clip(scr.astype(np.float32) / 255.0, 0.0, 1.0) ** 1.04
        return np.clip(scr_f * 255.0, 0.0, 255.0).astype(np.uint8)

    if mode == "upscale_1_8x":
        return cv2.resize(img, (int(round(w * 1.8)), int(round(h * 1.8))), interpolation=cv2.INTER_CUBIC)

    if mode == "downscale_0_68x_png":
        cropped = img[3 : max(67, h - 2), 5 : max(69, w - 2)] if (h > 96 and w > 96) else img
        ch, cw = cropped.shape[:2]
        return cv2.resize(
            cropped, (max(160, int(round(cw * 0.68))), max(160, int(round(ch * 0.68)))), interpolation=cv2.INTER_AREA
        )

    if mode == "jpeg_q96_444":
        cropped = img[5 : max(69, h - 3), 3 : max(67, w - 3)] if (h > 96 and w > 96) else img
        ch, cw = cropped.shape[:2]
        up = cv2.resize(cropped, (int(round(cw * 1.15)), int(round(ch * 1.15))), interpolation=cv2.INTER_LANCZOS4)
        return _encode_decode_rgb(up, "jpeg_444", 96)

    if mode == "phone_isp":
        smooth = cv2.bilateralFilter(img, d=5, sigmaColor=18, sigmaSpace=18)
        blur = cv2.GaussianBlur(smooth, (0, 0), 1.1)
        sharp = cv2.addWeighted(smooth, 1.28, blur, -0.28, 0)
        return _encode_decode_rgb(sharp, "jpeg_420", 90)

    if mode == "fft_hf_knockout":
        img_f = img.astype(np.float32) / 255.0
        fy = np.fft.fftfreq(h) * 2.0
        fx = np.fft.rfftfreq(w) * 2.0
        yy, xx = np.meshgrid(fy, fx, indexing="ij")
        rad = np.sqrt(xx * xx + yy * yy)
        notch = 1.0 - 0.85 * ((rad >= 0.50) & (rad <= 0.88)).astype(np.float32)
        out = np.empty_like(img_f)
        for ch_i in range(3):
            spec = np.fft.rfft2(img_f[:, :, ch_i])
            out[:, :, ch_i] = np.fft.irfft2(spec * notch, s=(h, w))
        return np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)

    raise ValueError(f"Unknown perturbation mode: {mode}")


class BalancedMultiGenPairDataset(Dataset):
    """
    Multi-Generator & Internet-Laundered Stratified Pair Dataset (`ArtifactNet v5`):
      - Pairs diverse Real camera domains (Pexels, LAION, MS-COCO, Smartphone VISION/Forchheim/IMD2020/FFHQ,
        Reddit photography, Shutterstock) with stratified 11 AI Generator Families (65+ models).
      - Caps legacy single-generator `legacy_train_v1` rows to <= 12% of epoch pairs so modern multi-generator
        and internet-circulated corpora receive >= 88% of training capacity while rotating through 100% of data across epochs.
      - Applies the EXACT same `sample_internet_laundering_params` pipeline to `(Real_k, AI_k)` in every pair,
        eliminating compression, scale, messenger, and screenshot shortcuts!
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        img_size: int = 256,
        pairs_per_epoch: int = 44000,
        seed: int = 42,
    ) -> None:
        self.data_dir = data_dir
        self.img_size = img_size
        self.base_seed = seed

        work = df.copy()
        if "family" not in work.columns:
            work["family"] = np.where(work["label"].astype(int) == 0, 0, 11)
        if "source_dataset" not in work.columns:
            work["source_dataset"] = "multigen"

        is_legacy = work["source_dataset"].astype(str) == "legacy_train_v1"
        mod_df = work[~is_legacy]
        leg_df = work[is_legacy]

        # Modern Real buckets grouped by source_dataset so smartphone/COCO/Reddit/Pexels all get strong representation
        self.mod_real_buckets: list[list[str]] = []
        mod_real = mod_df[mod_df["label"] == 0]
        if len(mod_real) > 0:
            for _, grp in mod_real.groupby("source_dataset", sort=True):
                files = grp["file_name"].astype(str).tolist()
                if files:
                    self.mod_real_buckets.append(files)

        # Modern AI buckets grouped by generator family (1..11)
        self.mod_ai_buckets: list[tuple[int, list[str]]] = []
        mod_ai = mod_df[mod_df["label"] == 1]
        if len(mod_ai) > 0:
            for fam_id, grp in mod_ai.groupby("family", sort=True):
                files = grp["file_name"].astype(str).tolist()
                if files:
                    self.mod_ai_buckets.append((int(fam_id), files))

        self.leg_real_files = leg_df[leg_df["label"] == 0]["file_name"].astype(str).tolist()
        self.leg_ai_files = leg_df[leg_df["label"] == 1]["file_name"].astype(str).tolist()

        total_real = sum(len(b) for b in self.mod_real_buckets) + len(self.leg_real_files)
        total_ai = sum(len(b[1]) for b in self.mod_ai_buckets) + len(self.leg_ai_files)
        max_natural = max(1, min(total_real, total_ai))
        self.pairs_per_epoch = pairs_per_epoch if pairs_per_epoch > 0 else max_natural

        self.epoch_real: list[str] = []
        self.epoch_ai: list[str] = []
        self.epoch_ai_fam: list[int] = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        rng = np.random.default_rng(self.base_seed + int(epoch) * 10007)
        n_pairs = self.pairs_per_epoch

        has_modern = len(self.mod_real_buckets) > 0 and len(self.mod_ai_buckets) > 0
        has_legacy = len(self.leg_real_files) > 0 and len(self.leg_ai_files) > 0

        if has_modern and has_legacy:
            n_leg = min(int(round(n_pairs * 0.12)), len(self.leg_real_files), len(self.leg_ai_files))
            n_mod = n_pairs - n_leg
        elif has_modern:
            n_mod, n_leg = n_pairs, 0
        else:
            n_mod, n_leg = 0, n_pairs

        real_out: list[str] = []
        ai_out: list[str] = []
        fam_out: list[int] = []

        if n_mod > 0:
            # Square-root damped bucket weights so smaller camera/generator families get boosted without extreme oversampling
            r_weights = np.array([max(1.0, len(b) ** 0.65) for b in self.mod_real_buckets], dtype=np.float64)
            r_weights /= r_weights.sum()
            r_counts = rng.multinomial(n_mod, r_weights)
            for b_files, cnt in zip(self.mod_real_buckets, r_counts):
                if cnt <= 0:
                    continue
                idx = rng.choice(len(b_files), size=int(cnt), replace=(cnt > len(b_files)))
                real_out.extend(b_files[i] for i in idx)

            a_weights = np.array([max(1.0, len(b[1]) ** 0.65) for b in self.mod_ai_buckets], dtype=np.float64)
            a_weights /= a_weights.sum()
            a_counts = rng.multinomial(n_mod, a_weights)
            for (fam_id, b_files), cnt in zip(self.mod_ai_buckets, a_counts):
                if cnt <= 0:
                    continue
                idx = rng.choice(len(b_files), size=int(cnt), replace=(cnt > len(b_files)))
                ai_out.extend(b_files[i] for i in idx)
                fam_out.extend([fam_id] * int(cnt))

            # Shuffle modern reals & modern AIs independently so every Real domain pairs with every AI family
            perm_r = rng.permutation(len(real_out))
            perm_a = rng.permutation(len(ai_out))
            real_out = [real_out[i] for i in perm_r]
            ai_out = [ai_out[i] for i in perm_a]
            fam_out = [fam_out[i] for i in perm_a]

        if n_leg > 0:
            n_avail = min(len(self.leg_real_files), len(self.leg_ai_files))
            l_idx = rng.choice(n_avail, size=n_leg, replace=(n_leg > n_avail))
            for i in l_idx:
                real_out.append(self.leg_real_files[int(i)])
                ai_out.append(self.leg_ai_files[int(i)])
                fam_out.append(11)

        # Final shuffle across all pairs for the epoch
        order = rng.permutation(len(real_out))
        self.epoch_real = [real_out[i] for i in order]
        self.epoch_ai = [ai_out[i] for i in order]
        self.epoch_ai_fam = [fam_out[i] for i in order]

    def __len__(self) -> int:
        return len(self.epoch_real)

    def _read_rgb(self, rel_path: str) -> np.ndarray:
        bgr = cv2.imread(str(self.data_dir / rel_path), cv2.IMREAD_COLOR)
        if bgr is None:
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int):
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        rng = np.random.default_rng(seed)
        aug_cfg = sample_internet_laundering_params(rng)

        raw_real = self._read_rgb(self.epoch_real[idx])
        raw_ai = self._read_rgb(self.epoch_ai[idx])
        ai_fam = self.epoch_ai_fam[idx]

        aug_real = apply_internet_laundering(raw_real, aug_cfg)
        aug_ai = apply_internet_laundering(raw_ai, aug_cfg)

        xg_r, xr_r, xp_r, _ = prepare_forensic_views(
            aug_real,
            img_size=self.img_size,
            strip_borders=False,
            is_train=True,
            rng=np.random.default_rng(aug_cfg["patch_seed"]),
        )
        xg_a, xr_a, xp_a, _ = prepare_forensic_views(
            aug_ai,
            img_size=self.img_size,
            strip_borders=False,
            is_train=True,
            rng=np.random.default_rng(aug_cfg["patch_seed"] ^ 0x5A5A5A),
        )

        if aug_cfg["do_mask"]:
            xr_r = apply_patch_mask_sync(xr_r, aug_cfg["mask_count"], aug_cfg["mask_seed"])
            xr_a = apply_patch_mask_sync(xr_a, aug_cfg["mask_count"], aug_cfg["mask_seed"])

        xg_pair = torch.from_numpy(np.stack([xg_r, xg_a], axis=0).transpose(0, 3, 1, 2))
        xr_pair = torch.from_numpy(np.stack([xr_r, xr_a], axis=0).transpose(0, 3, 1, 2))
        xp_pair = torch.from_numpy(np.stack([xp_r, xp_a], axis=0).transpose(0, 3, 1, 2))
        t_pair = torch.tensor([0.0, 1.0], dtype=torch.float32)
        fam_pair = torch.tensor([0, int(ai_fam)], dtype=torch.long)
        return xg_pair, xr_pair, xp_pair, t_pair, fam_pair


class PairedForensicDataset(BalancedMultiGenPairDataset):
    """Backward-compatible wrapper around BalancedMultiGenPairDataset."""

    def __init__(self, df: pd.DataFrame, data_dir: Path, img_size: int = 256) -> None:
        n_default = max(1, min(int((df["label"] == 0).sum()), int((df["label"] == 1).sum())))
        super().__init__(df=df, data_dir=data_dir, img_size=img_size, pairs_per_epoch=n_default)


def collate_paired_batch(batch):
    xg = torch.cat([item[0] for item in batch], dim=0)
    xr = torch.cat([item[1] for item in batch], dim=0)
    xp = torch.cat([item[2] for item in batch], dim=0)
    targets = torch.cat([item[3] for item in batch], dim=0)
    if len(batch[0]) >= 5:
        families = torch.cat([item[4] for item in batch], dim=0)
        return xg, xr, xp, targets, families
    return xg, xr, xp, targets


class SingleForensicDataset(Dataset):
    """Deterministic Evaluation / Perturbation Dataset returning (x_global, x_rich, x_poor, target[, family])."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        img_size: int = 256,
        perturbation: str = "clean",
        return_global: bool = True,
        return_family: bool = False,
    ) -> None:
        self.file_names = df["file_name"].astype(str).tolist()
        self.labels = df["label"].astype(np.float32).tolist()
        if "family" in df.columns:
            self.families = df["family"].fillna(0).astype(np.int64).tolist()
        else:
            self.families = [0 if lbl < 0.5 else 11 for lbl in self.labels]
        self.data_dir = data_dir
        self.img_size = img_size
        self.perturbation = perturbation
        self.return_global = return_global
        self.return_family = return_family

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int):
        img_path = self.data_dir / self.file_names[idx]
        label = self.labels[idx]

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            rgb = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self.perturbation != "clean":
            rgb = apply_eval_perturbation(rgb, self.perturbation)

        xg_np, xr_np, xp_np, _ = prepare_forensic_views(
            rgb, img_size=self.img_size, strip_borders=True, is_train=False
        )
        xg = torch.from_numpy(np.ascontiguousarray(xg_np.transpose(2, 0, 1)))
        xr = torch.from_numpy(np.ascontiguousarray(xr_np.transpose(2, 0, 1)))
        xp = torch.from_numpy(np.ascontiguousarray(xp_np.transpose(2, 0, 1)))
        target = torch.tensor(label, dtype=torch.float32)

        if self.return_family:
            fam = torch.tensor(self.families[idx], dtype=torch.long)
            if self.return_global:
                return xg, xr, xp, target, fam
            return xr, xp, target, fam

        if self.return_global:
            return xg, xr, xp, target
        return xr, xp, target


ForensicPairDataset = SingleForensicDataset


def create_dataloaders_v5(
    data_dir: Path,
    img_size: int = 256,
    batch_size: int = 32,
    samples_per_epoch: int = 88000,
    num_workers: int = 8,
    seed: int = 42,
) -> tuple[BalancedMultiGenPairDataset, DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Creates v5 Multi-Generator + Unseen-OOD DataLoaders:
      Returns (train_ds, train_loader, val_id_loader, val_ood_loader, test_ood_loader)
    """
    train_df, val_id_df, val_ood_df, test_ood_df = prepare_splits_v5(data_dir)
    pair_batch_size = max(1, batch_size // 2)
    pairs_per_epoch = max(pair_batch_size, samples_per_epoch // 2)

    train_kw = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=3 if num_workers > 0 else None,
    )
    eval_workers = max(2, num_workers // 2) if num_workers > 0 else 0
    eval_kw = dict(
        num_workers=eval_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=eval_workers > 0,
        prefetch_factor=2 if eval_workers > 0 else None,
    )

    train_ds = BalancedMultiGenPairDataset(
        train_df, data_dir, img_size=img_size, pairs_per_epoch=pairs_per_epoch, seed=seed
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=pair_batch_size,
        shuffle=False,  # BalancedMultiGenPairDataset.set_epoch() performs stratified pair shuffling!
        drop_last=True,
        collate_fn=collate_paired_batch,
        **train_kw,
    )

    val_id_ds = SingleForensicDataset(val_id_df, data_dir, img_size=img_size, return_global=True, return_family=True)
    val_ood_ds = SingleForensicDataset(val_ood_df, data_dir, img_size=img_size, return_global=True, return_family=True)
    test_ood_ds = SingleForensicDataset(test_ood_df, data_dir, img_size=img_size, return_global=True, return_family=True)

    eval_bs = max(batch_size, 48)
    val_id_loader = DataLoader(val_id_ds, batch_size=eval_bs, shuffle=False, **eval_kw)
    val_ood_loader = DataLoader(val_ood_ds, batch_size=eval_bs, shuffle=False, **eval_kw)
    test_ood_loader = DataLoader(test_ood_ds, batch_size=eval_bs, shuffle=False, **eval_kw)

    return train_ds, train_loader, val_id_loader, val_ood_loader, test_ood_loader


def create_dataloaders(
    data_dir: Path,
    img_size: int = 256,
    batch_size: int = 64,
    num_workers: int = 6,
    return_global: bool = True,
    paired_train: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Legacy single-dataset dataloader builder preserved for backward compatibility."""
    train_df, val_df, test_df = prepare_splits(data_dir)

    loader_kw = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=3 if num_workers > 0 else None,
    )

    if paired_train and return_global:
        pair_batch_size = max(1, batch_size // 2)
        train_ds = PairedForensicDataset(train_df, data_dir, img_size=img_size)
        train_loader = DataLoader(
            train_ds,
            batch_size=pair_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_paired_batch,
            **loader_kw,
        )
    else:
        train_ds = SingleForensicDataset(
            train_df, data_dir, img_size=img_size, return_global=return_global
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kw)

    val_ds = SingleForensicDataset(val_df, data_dir, img_size=img_size, return_global=return_global)
    test_ds = SingleForensicDataset(test_df, data_dir, img_size=img_size, return_global=return_global)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kw)
    return train_loader, val_loader, test_loader
