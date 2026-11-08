from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


CANONICAL_MAX_LONG_SIDE = 896


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
        # Composite RGBA onto neutral mid-gray so transparent borders don't create harsh step edges
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
    across 768px dataset images, 1792px generator outputs, and 4000px camera photos.
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

    # Supplement with half-stride shifted patches when source resolution is compact
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


def sample_shared_aug_params(rng: np.random.Generator) -> dict:
    """
    Samples a single random imaging-pipeline configuration applied IDENTICALLY to both
    Real_k and AI_k in a matched pair, cancelling all compression, scale, and ISP shortcuts.
    """
    interps = [cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
    codec_roll = float(rng.random())
    if codec_roll < 0.30:
        codec = "jpeg_444"
    elif codec_roll < 0.60:
        codec = "jpeg_420"
    elif codec_roll < 0.68:
        codec = "webp"
    else:
        codec = "none"  # Uncompressed float->uint8 pass-through (mimics PNG uploads)

    isp_roll = float(rng.random())
    if isp_roll < 0.16:
        isp_mode = "sharpen"
    elif isp_roll < 0.28:
        isp_mode = "soften"
    elif isp_roll < 0.42:
        isp_mode = "iso_grain"
    else:
        isp_mode = "none"

    return {
        # 1. Sub-8x8 pixel crop offset breaks baked-in Q75 DCT block alignment
        "do_grid_shift": bool(rng.random() < 0.85),
        "dy1": int(rng.integers(0, 8)),
        "dx1": int(rng.integers(0, 8)),
        "dy2": int(rng.integers(0, 8)),
        "dx2": int(rng.integers(0, 8)),
        # 2. Exact dihedral symmetries
        "hflip": bool(rng.random() < 0.50),
        "vflip": bool(rng.random() < 0.20),
        "rot90_k": int(rng.integers(0, 4)) if rng.random() < 0.50 else 0,
        # 3. Multi-scale resampling (both <1.0 downscale and >1.0 upscale!)
        "do_scale": bool(rng.random() < 0.68),
        "scale": float(rng.uniform(0.56, 1.38)),
        "interp": int(interps[int(rng.integers(0, len(interps)))]),
        # 4. Symmetric ISP / optical perturbation (never flips class label!)
        "isp_mode": isp_mode,
        "sharpen_alpha": float(rng.uniform(0.15, 0.42)),
        "blur_sigma": float(rng.uniform(0.35, 0.90)),
        "grain_sigma": float(rng.uniform(1.2, 4.5)),
        "grain_seed": int(rng.integers(0, 2**31 - 1)),
        # 5. Photometric & monochrome equivariance
        "do_photo": bool(rng.random() < 0.35),
        "alpha_gain": float(rng.uniform(0.90, 1.10)),
        "beta_bias": float(rng.uniform(-10.0, 10.0)),
        "gamma": float(rng.uniform(0.90, 1.12)),
        "to_monochrome": bool(rng.random() < 0.06),
        # 6. Multi-codec & chroma-subsampling diversity
        "codec": codec,
        "quality": int(rng.integers(52, 99)),
        # 7. Synchronized rich-patch masking
        "do_mask": bool(rng.random() < 0.30),
        "mask_count": int(rng.integers(2, 11)),
        "mask_seed": int(rng.integers(0, 2**31 - 1)),
        "patch_seed": int(rng.integers(0, 2**31 - 1)),
    }


def apply_shared_aug(img_rgb: np.ndarray, p: dict) -> np.ndarray:
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

    if p["do_scale"]:
        ch, cw = out.shape[:2]
        nh = int(np.clip(round(ch * p["scale"]), 192, 1024))
        nw = int(np.clip(round(cw * p["scale"]), 192, 1024))
        out = cv2.resize(out, (nw, nh), interpolation=p["interp"])

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

    codec = p["codec"]
    if codec != "none":
        bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        ok = False
        enc = None
        if codec == "jpeg_444":
            params = [
                int(cv2.IMWRITE_JPEG_QUALITY),
                p["quality"],
                int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR),
                int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444),
            ]
            ok, enc = cv2.imencode(".jpg", bgr, params)
        elif codec == "jpeg_420":
            q = max(48, min(95, p["quality"]))
            params = [
                int(cv2.IMWRITE_JPEG_QUALITY),
                q,
                int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR),
                int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420),
            ]
            ok, enc = cv2.imencode(".jpg", bgr, params)
        elif codec == "webp":
            ok, enc = cv2.imencode(".webp", bgr, [int(cv2.IMWRITE_WEBP_QUALITY), max(58, p["quality"])])

        if ok and enc is not None:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                out = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

    return out


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
    Deterministic out-of-domain imaging transforms for epoch validation & final stress-testing:
    - 'clean': untouched dataset image
    - 'upscale_1_8x': 1.8x bicubic upscale (breaks 768px pixel pitch; caught v2's 8% Real specificity bug)
    - 'downscale_0_68x_png': (3,5) sub-block shift + 0.68x INTER_AREA (erases Q75 8x8 grid, lossless PNG-like)
    - 'jpeg_q96_444': (5,3) shift + 1.15x Lanczos + JPEG Q96 4:4:4 (mimics modern DSLR & OOD generator JPEGs)
    - 'phone_isp': bilateral smooth + unsharp mask + JPEG Q90 (mimics iPhone/Samsung computational ISP)
    - 'fft_hf_knockout': notches out radial high-frequency ring [0.50, 0.88] in 2D FFT (tests shortcut removal)
    """
    img = ensure_rgb_uint8(img_rgb)
    h, w = img.shape[:2]

    if mode == "clean":
        return img
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
        bgr = cv2.cvtColor(up, cv2.COLOR_RGB2BGR)
        ok, enc = cv2.imencode(
            ".jpg",
            bgr,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                96,
                int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR),
                int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444),
            ],
        )
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
        return up
    if mode == "phone_isp":
        smooth = cv2.bilateralFilter(img, d=5, sigmaColor=18, sigmaSpace=18)
        blur = cv2.GaussianBlur(smooth, (0, 0), 1.1)
        sharp = cv2.addWeighted(smooth, 1.28, blur, -0.28, 0)
        bgr = cv2.cvtColor(sharp, cv2.COLOR_RGB2BGR)
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
        return sharp
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


class PairedForensicDataset(Dataset):
    """
    Pair-Synchronized Training Dataset:
    Each item returns a matched (Real_k, AI_k) semantic pair processed through the EXACT same
    random crop-offset, scale/interpolation, ISP, and codec augmentation pipeline.
    When collated via `collate_paired_batch`, batch indices `0::2` are Real (0.0) and `1::2` are AI (1.0).
    """

    def __init__(self, df: pd.DataFrame, data_dir: Path, img_size: int = 256) -> None:
        self.data_dir = data_dir
        self.img_size = img_size

        if "pair_id" in df.columns:
            real_df = df[df["label"] == 0].set_index("pair_id")["file_name"]
            ai_df = df[df["label"] == 1].set_index("pair_id")["file_name"]
            common_pairs = real_df.index.intersection(ai_df.index)
            self.real_files = real_df.loc[common_pairs].astype(str).tolist()
            self.ai_files = ai_df.loc[common_pairs].astype(str).tolist()
        else:
            reals = df[df["label"] == 0]["file_name"].astype(str).tolist()
            ais = df[df["label"] == 1]["file_name"].astype(str).tolist()
            n = min(len(reals), len(ais))
            self.real_files = reals[:n]
            self.ai_files = ais[:n]

    def __len__(self) -> int:
        return len(self.real_files)

    def _read_rgb(self, rel_path: str) -> np.ndarray:
        bgr = cv2.imread(str(self.data_dir / rel_path), cv2.IMREAD_COLOR)
        if bgr is None:
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int):
        # Worker-safe per-sample seed drawn from PyTorch's worker RNG (unique across workers & epochs)
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        rng = np.random.default_rng(seed)
        aug_cfg = sample_shared_aug_params(rng)

        raw_real = self._read_rgb(self.real_files[idx])
        raw_ai = self._read_rgb(self.ai_files[idx])

        aug_real = apply_shared_aug(raw_real, aug_cfg)
        aug_ai = apply_shared_aug(raw_ai, aug_cfg)

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
        return xg_pair, xr_pair, xp_pair, t_pair


def collate_paired_batch(batch):
    xg = torch.cat([item[0] for item in batch], dim=0)
    xr = torch.cat([item[1] for item in batch], dim=0)
    xp = torch.cat([item[2] for item in batch], dim=0)
    targets = torch.cat([item[3] for item in batch], dim=0)
    return xg, xr, xp, targets


class SingleForensicDataset(Dataset):
    """Deterministic Evaluation / Perturbation Dataset returning (x_global, x_rich, x_poor, target)."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        img_size: int = 256,
        perturbation: str = "clean",
        return_global: bool = True,
    ) -> None:
        self.file_names = df["file_name"].astype(str).tolist()
        self.labels = df["label"].astype(np.float32).tolist()
        self.data_dir = data_dir
        self.img_size = img_size
        self.perturbation = perturbation
        self.return_global = return_global

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

        if self.return_global:
            return xg, xr, xp, target
        return xr, xp, target


# Backward-compatible alias
ForensicPairDataset = SingleForensicDataset


def create_dataloaders(
    data_dir: Path,
    img_size: int = 256,
    batch_size: int = 64,
    num_workers: int = 6,
    return_global: bool = True,
    paired_train: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
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
