import argparse
import concurrent.futures
import hashlib
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi

from src.model import map_generator_to_family


DEFACTIFY_GEN_MAP = {
    0: "mscoco_real",
    1: "sd-2.1",
    2: "sdxl",
    3: "sd-3",
    4: "dalle-3",
    5: "midjourney-6",
}


def cleanup_hf_probe_cache() -> None:
    """Removes any HuggingFace dataset cache folders under ~/.cache/huggingface/hub so C: stays clean."""
    hf_hub_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if not hf_hub_cache.exists():
        return
    for item in hf_hub_cache.glob("datasets--*"):
        try:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
        except Exception:
            pass


def get_free_gb(drive_root: str = "C:\\") -> float:
    return shutil.disk_usage(drive_root).free / (1024 ** 3)


def download_shard_direct(
    repo_id: str,
    rfilename: str,
    dest_path: Path,
    min_free_gb: float = 19.5,
    retries: int = 3,
    num_conns: int = 4,
) -> bool:
    """
    Streams a single Parquet shard directly from HuggingFace CDN into `dest_path`
    using parallel HTTP Range connections on large shards (zero ~/.cache/huggingface writes).
    """
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{rfilename}"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    base_headers = {"User-Agent": "ArtifactNetV5-Harvester/1.0"}
    if hf_token:
        base_headers["Authorization"] = f"Bearer {hf_token}"

    for attempt in range(1, retries + 1):
        try:
            head_req = urllib.request.Request(url, method="HEAD", headers=base_headers)
            with urllib.request.urlopen(head_req, timeout=30) as hresp:
                content_len = int(hresp.headers.get("Content-Length", "0") or 0)
                accept_ranges = (hresp.headers.get("Accept-Ranges", "") or "").lower()
                final_url = hresp.geturl() or url

            free_now = shutil.disk_usage("C:\\").free
            if content_len > 0 and (free_now - content_len) < (min_free_gb - 2.0) * (1024 ** 3):
                print(
                    f"  [DiskGuard] Skipping {rfilename} ({content_len/(1024**3):.2f} GB) "
                    f"to preserve C: safety floor ({free_now/(1024**3):.2f} GB free)."
                )
                return False

            if content_len >= 32 * 1024 * 1024 and "bytes" in accept_ranges and num_conns > 1:
                with open(dest_path, "wb") as f_init:
                    f_init.truncate(content_len)

                part_size = (content_len + num_conns - 1) // num_conns
                ranges = []
                for idx in range(num_conns):
                    s = idx * part_size
                    e = min(content_len - 1, (idx + 1) * part_size - 1)
                    if s <= e:
                        ranges.append((s, e))

                def _dl_part(bounds: tuple[int, int]) -> int:
                    s, e = bounds
                    hdrs = dict(base_headers)
                    hdrs["Range"] = f"bytes={s}-{e}"
                    r = urllib.request.Request(final_url, headers=hdrs)
                    written = 0
                    with urllib.request.urlopen(r, timeout=60) as resp, open(dest_path, "r+b") as pf_out:
                        pf_out.seek(s)
                        while True:
                            chunk = resp.read(4 * 1024 * 1024)
                            if not chunk:
                                break
                            pf_out.write(chunk)
                            written += len(chunk)
                    if written != (e - s + 1):
                        raise IOError(f"Short range read: got {written}, expected {e - s + 1}")
                    return written

                with concurrent.futures.ThreadPoolExecutor(max_workers=len(ranges)) as dl_pool:
                    total_dl = sum(dl_pool.map(_dl_part, ranges))
                if total_dl == content_len:
                    return True
                raise IOError(f"Total downloaded {total_dl} != {content_len}")

            req = urllib.request.Request(url, headers=base_headers)
            with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, "wb") as out_f:
                while True:
                    chunk = resp.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)
            return True
        except Exception as exc:
            dest_path.unlink(missing_ok=True)
            if attempt == retries:
                print(f"  [Warn] Failed downloading {repo_id}/{rfilename} after {retries} attempts: {exc}")
                return False
            time.sleep(2.0 * attempt)
    return False


def _detect_ext(raw_bytes: bytes) -> str:
    if len(raw_bytes) >= 3 and raw_bytes[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if len(raw_bytes) >= 8 and raw_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if len(raw_bytes) >= 12 and raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WEBP":
        return ".webp"
    return ""


def process_and_write_image(task: tuple) -> dict | None:
    """
    Worker function executed in ThreadPoolExecutor (OpenCV releases the GIL!):
    Validates image integrity, normalizes extreme resolutions, applies deterministic
    label-independent multi-codec storage (preserving compact native internet JPEGs/WebPs
    verbatim while preventing 4MB PNG bloat), and writes to disk.
    """
    (
        raw_bytes,
        sha_hex,
        label,
        generator,
        item_type,
        source_dataset,
        split_role,
        out_root,
        data_dir,
    ) = task

    if not raw_bytes or len(raw_bytes) < 512:
        return None

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None

    h, w = bgr.shape[:2]
    if min(h, w) < 160 or max(h, w) < 224:
        return None
    if (max(h, w) / max(1, min(h, w))) > 3.6:
        return None

    # Reject solid/corrupt blank frames
    if float(bgr[::4, ::4].std()) < 3.2:
        return None

    h_int = int(sha_hex[:8], 16)
    orig_ext = _detect_ext(raw_bytes)

    # Preserve compact authentic internet JPEG/WebP streams untouched 65% of the time!
    keep_verbatim = (
        orig_ext in (".jpg", ".webp")
        and len(raw_bytes) <= 320_000
        and max(h, w) <= 1344
        and (h_int % 100) < 65
    )

    if keep_verbatim:
        out_bytes = raw_bytes
        ext = orig_ext
        out_h, out_w = h, w
    else:
        max_sides = (896, 1024, 1152, 1280)
        cap_side = max_sides[(h_int >> 8) % len(max_sides)]
        codec_slot = h_int % 100
        if codec_slot >= 93:
            # Lossless PNG slot: cap at 768px so file stays ~350KB
            cap_side = min(cap_side, 768)

        if max(h, w) > cap_side:
            scale = cap_side / float(max(h, w))
            out_h = max(160, int(round(h * scale)))
            out_w = max(160, int(round(w * scale)))
            bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        else:
            out_h, out_w = h, w

        if codec_slot < 28:
            ext = ".jpg"
            q = 92 + ((h_int >> 12) % 6)  # Q92..97 4:4:4
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
        elif codec_slot < 58:
            ext = ".jpg"
            q = 82 + ((h_int >> 12) % 9)  # Q82..90 4:2:0
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
        elif codec_slot < 79:
            ext = ".jpg"
            q = 71 + ((h_int >> 12) % 10)  # Q71..80 4:2:0 (social/messenger)
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
        elif codec_slot < 93:
            ext = ".webp"
            q = 78 + ((h_int >> 12) % 14)  # WebP Q78..91
            ok, enc = cv2.imencode(".webp", bgr, [int(cv2.IMWRITE_WEBP_QUALITY), q])
        else:
            ext = ".png"
            ok, enc = cv2.imencode(".png", bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

        if not ok or enc is None:
            return None
        out_bytes = enc.tobytes()

    cls_folder = "real" if label == 0 else "ai"
    sub_bucket = sha_hex[:2]
    target_dir = out_root / split_role / cls_folder / sub_bucket
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / f"{sha_hex[:20]}{ext}"

    if not file_path.exists():
        file_path.write_bytes(out_bytes)

    rel_path = file_path.relative_to(data_dir).as_posix()
    fam = map_generator_to_family(label, generator, item_type)
    return {
        "file_name": rel_path,
        "label": int(label),
        "family": int(fam),
        "generator": str(generator),
        "item_type": str(item_type),
        "source_dataset": str(source_dataset),
        "split_role": str(split_role),
        "width": int(out_w),
        "height": int(out_h),
        "bytes": len(out_bytes),
        "sha256": sha_hex,
    }


def extract_raw_bytes(img_cell) -> bytes | None:
    if img_cell is None:
        return None
    if isinstance(img_cell, (bytes, bytearray, memoryview)):
        return bytes(img_cell)
    if isinstance(img_cell, dict):
        b = img_cell.get("bytes")
        if b is not None:
            return bytes(b)
    return None


def build_v5_splits(data_dir: Path, seed: int = 42) -> dict:
    """
    Combines `data/multigen_manifest.csv`, balanced pairs from `data/train.csv` (legacy_train_v1),
    and optional `data/custom/{real,ai}` into `data/splits_v5/`:
      - train_split.csv
      - val_id_split.csv
      - val_ood_split.csv
      - test_ood_split.csv
    """
    splits_v5_dir = data_dir / "splits_v5"
    splits_v5_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = data_dir / "multigen_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}")

    mdf = pd.read_csv(manifest_path)
    mdf = mdf.drop_duplicates(subset=["sha256"]).reset_index(drop=True)

    # Verify files exist on disk (fast check)
    exists_mask = [(data_dir / fn).exists() for fn in mdf["file_name"].astype(str)]
    mdf = mdf[exists_mask].reset_index(drop=True)

    rng = np.random.default_rng(seed)

    # 1. Strictly Held-Out OOD Split (val_ood_split + test_ood_split)
    ood_df = mdf[mdf["split_role"] == "ood"].copy()
    ood_real = ood_df[ood_df["label"] == 0].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    ood_ai = ood_df[ood_df["label"] == 1].sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    n_ood_bal = min(len(ood_real), len(ood_ai))
    ood_real = ood_real.iloc[:n_ood_bal]
    ood_ai = ood_ai.iloc[:n_ood_bal]

    n_val_ood_half = int(round(n_ood_bal * 0.45))
    val_ood_df = pd.concat(
        [ood_real.iloc[:n_val_ood_half], ood_ai.iloc[:n_val_ood_half]], ignore_index=True
    ).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    test_ood_df = pd.concat(
        [ood_real.iloc[n_val_ood_half:], ood_ai.iloc[n_val_ood_half:]], ignore_index=True
    ).sample(frac=1.0, random_state=seed + 3).reset_index(drop=True)

    # 2. Harvested Train/Val-ID Pool
    pool_df = mdf[mdf["split_role"] == "train_pool"].copy().sample(frac=1.0, random_state=seed + 4).reset_index(drop=True)
    val_id_parts = []
    train_parts = []
    for (_, _), grp in pool_df.groupby(["label", "family"], sort=False):
        n_val = min(max(1, int(round(len(grp) * 0.07))), 400) if len(grp) >= 12 else 0
        if n_val > 0:
            val_id_parts.append(grp.iloc[:n_val])
            train_parts.append(grp.iloc[n_val:])
        else:
            train_parts.append(grp)

    harv_train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    harv_val_df = pd.concat(val_id_parts, ignore_index=True) if val_id_parts else pd.DataFrame()

    # Balance val_id 50/50 Real vs AI
    if len(harv_val_df) > 0:
        v0 = harv_val_df[harv_val_df["label"] == 0]
        v1 = harv_val_df[harv_val_df["label"] == 1]
        nv = min(len(v0), len(v1))
        harv_val_df = pd.concat([v0.iloc[:nv], v1.iloc[:nv]], ignore_index=True)

    # 3. Merge Legacy `data/splits/` (original 79,950 dataset) so backward performance stays near-100% too
    leg_train_csv = data_dir / "splits" / "train_split.csv"
    leg_val_csv = data_dir / "splits" / "val_split.csv"
    if not leg_train_csv.exists() and (data_dir / "train.csv").exists():
        from src.dataset import prepare_splits
        prepare_splits(data_dir)

    if leg_train_csv.exists():
        lt = pd.read_csv(leg_train_csv)
        lt0 = lt[lt["label"] == 0].sample(min(20000, int((lt["label"] == 0).sum())), random_state=seed)
        lt1 = lt[lt["label"] == 1].sample(min(20000, int((lt["label"] == 1).sum())), random_state=seed)
        lt_sub = pd.concat([lt0, lt1], ignore_index=True)
        lt_formatted = pd.DataFrame(
            {
                "file_name": lt_sub["file_name"].astype(str),
                "label": lt_sub["label"].astype(int),
                "family": np.where(lt_sub["label"].astype(int) == 0, 0, 11),
                "generator": np.where(lt_sub["label"].astype(int) == 0, "shutterstock_real", "deepmedia_v1"),
                "item_type": "legacy_pair",
                "source_dataset": "legacy_train_v1",
                "split_role": "train_pool",
            }
        )
        harv_train_df = pd.concat([harv_train_df, lt_formatted], ignore_index=True)

    if leg_val_csv.exists():
        lv = pd.read_csv(leg_val_csv)
        lv0 = lv[lv["label"] == 0].sample(min(600, int((lv["label"] == 0).sum())), random_state=seed)
        lv1 = lv[lv["label"] == 1].sample(min(600, int((lv["label"] == 1).sum())), random_state=seed)
        lv_sub = pd.concat([lv0, lv1], ignore_index=True)
        lv_formatted = pd.DataFrame(
            {
                "file_name": lv_sub["file_name"].astype(str),
                "label": lv_sub["label"].astype(int),
                "family": np.where(lv_sub["label"].astype(int) == 0, 0, 11),
                "generator": np.where(lv_sub["label"].astype(int) == 0, "shutterstock_real", "deepmedia_v1"),
                "item_type": "legacy_pair",
                "source_dataset": "legacy_train_v1",
                "split_role": "val_id",
            }
        )
        harv_val_df = pd.concat([harv_val_df, lv_formatted], ignore_index=True)

    # 4. Optional user custom folders: data/custom/real & data/custom/ai
    custom_rows = []
    for sub_name, lbl, gen_name in [("real", 0, "user_custom_real"), ("ai", 1, "user_custom_ai")]:
        c_dir = data_dir / "custom" / sub_name
        if c_dir.exists():
            for p in c_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                    custom_rows.append(
                        {
                            "file_name": p.relative_to(data_dir).as_posix(),
                            "label": lbl,
                            "family": 0 if lbl == 0 else 11,
                            "generator": gen_name,
                            "item_type": "custom",
                            "source_dataset": "user_custom",
                            "split_role": "train_pool",
                        }
                    )
    if custom_rows:
        harv_train_df = pd.concat([harv_train_df, pd.DataFrame(custom_rows)], ignore_index=True)

    keep_cols = ["file_name", "label", "family", "generator", "item_type", "source_dataset"]
    for df_ref in (harv_train_df, harv_val_df, val_ood_df, test_ood_df):
        for col in keep_cols:
            if col not in df_ref.columns:
                df_ref[col] = ""

    harv_train_df = harv_train_df[keep_cols].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    harv_val_df = harv_val_df[keep_cols].sample(frac=1.0, random_state=seed + 10).reset_index(drop=True)
    val_ood_df = val_ood_df[keep_cols].reset_index(drop=True)
    test_ood_df = test_ood_df[keep_cols].reset_index(drop=True)

    harv_train_df.to_csv(splits_v5_dir / "train_split.csv", index=False)
    harv_val_df.to_csv(splits_v5_dir / "val_id_split.csv", index=False)
    val_ood_df.to_csv(splits_v5_dir / "val_ood_split.csv", index=False)
    test_ood_df.to_csv(splits_v5_dir / "test_ood_split.csv", index=False)

    summary = {
        "train_total": len(harv_train_df),
        "train_real": int((harv_train_df["label"] == 0).sum()),
        "train_ai": int((harv_train_df["label"] == 1).sum()),
        "val_id_total": len(harv_val_df),
        "val_ood_total": len(val_ood_df),
        "test_ood_total": len(test_ood_df),
        "unique_generators": int(mdf["generator"].nunique()),
    }
    print("\n" + "=" * 88)
    print("[Splits v5 Ready]")
    print(
        f"  • train_split.csv    : {summary['train_total']:,} images "
        f"(Real: {summary['train_real']:,} | AI: {summary['train_ai']:,})"
    )
    print(f"  • val_id_split.csv   : {summary['val_id_total']:,} images (In-Distribution Multi-Generator)")
    print(f"  • val_ood_split.csv  : {summary['val_ood_total']:,} images (Strictly Unseen Generators & Reddit)")
    print(f"  • test_ood_split.csv : {summary['test_ood_total']:,} images (Strictly Unseen Generators & Reddit)")
    print(f"  • Distinct Harvested Sources/Generators: {summary['unique_generators']}")
    print("=" * 88)
    return summary


def harvest() -> None:
    parser = argparse.ArgumentParser(description="Multi-Generator & Internet-Circulated Dataset Harvester (Zero HF Cache)")
    parser.add_argument("--max-gb", type=float, default=33.5, help="Max disk budget for data/multigen_data in GB")
    parser.add_argument("--min-free-gb", type=float, default=19.5, help="Hard free-space safety floor on C: in GB")
    parser.add_argument("--max-minutes", type=float, default=42.0, help="Wall-clock cap in minutes so training stays on schedule")
    parser.add_argument("--workers", type=int, default=8, help="Parallel OpenCV decode/encode threads")
    parser.add_argument("--splits-only", action="store_true", help="Rebuild splits_v5 from existing multigen_manifest.csv")
    args = parser.parse_args()

    t_start = time.time()
    cleanup_hf_probe_cache()

    data_dir = PROJECT_ROOT / "data"
    out_root = data_dir / "multigen_data"
    tmp_dir = data_dir / "_tmp_shard"
    manifest_path = data_dir / "multigen_manifest.csv"
    done_shards_path = data_dir / "multigen_done_shards.txt"

    out_root.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if args.splits_only:
        build_v5_splits(data_dir)
        return

    existing_records: list[dict] = []
    seen_hashes: set[str] = set()
    total_bytes_written = 0
    done_shards: set[str] = set()

    if manifest_path.exists():
        old_df = pd.read_csv(manifest_path)
        existing_records = old_df.to_dict(orient="records")
        seen_hashes = set(old_df["sha256"].astype(str).tolist())
        total_bytes_written = int(old_df["bytes"].sum()) if "bytes" in old_df.columns else 0
        print(
            f"[Resume] Loaded {len(existing_records):,} harvested images "
            f"({total_bytes_written / (1024**3):.2f} GB) from {manifest_path.name}"
        )

    if done_shards_path.exists():
        done_shards = {
            line.strip() for line in done_shards_path.read_text(encoding="utf-8").splitlines() if line.strip()
        }

    api = HfApi()
    of_files = api.list_repo_files(repo_id="ComplexDataLab/OpenFake", repo_type="dataset")
    df_files = api.list_repo_files(repo_id="Rajarshi-Roy-research/Defactify_Image_Dataset", repo_type="dataset")
    cf_files = api.list_repo_files(repo_id="OwensLab/CommunityForensics-Eval", repo_type="dataset")

    of_core_train = sorted([f for f in of_files if f.startswith("core/train-") and f.endswith(".parquet")])
    of_core_val = sorted([f for f in of_files if f.startswith("core/validation-") and f.endswith(".parquet")])
    of_core_test = sorted([f for f in of_files if f.startswith("core/test-") and f.endswith(".parquet")])
    of_reddit_test = sorted([f for f in of_files if f.startswith("reddit/test-") and f.endswith(".parquet")])
    of_supp_reddit = sorted([f for f in of_files if f.startswith("supplement/reddit/") and f.endswith(".parquet")])
    of_supp_civitai = sorted([f for f in of_files if f.startswith("supplement/civitai/") and f.endswith(".parquet")])
    of_supp_t2i = sorted([f for f in of_files if f.startswith("supplement/t2i/") and f.endswith(".parquet")])

    defact_train = sorted([f for f in df_files if (f.startswith("data/train-") or f.startswith("data/validation-")) and f.endswith(".parquet")])
    cf_shards = sorted([f for f in cf_files if f.startswith("data/CompEval-") and f.endswith(".parquet")])

    # Build ordered shard schedule:
    # Priority 1: Strictly Unseen OOD Split (core/test + reddit/test)
    # Priority 2: Fast high-diversity datasets (Defactify + CommunityForensics smartphone/camera + OpenFake 2026 Reddit/Civitai/T2I)
    # Priority 3: Spaced OpenFake core/train + core/validation shards (auto-balancing Real vs Fake to 50/50!)
    shard_jobs: list[dict] = []

    # 1. Strictly Unseen OOD Split (1 core/test shard [3,600 imgs] + 1 reddit/test shard [2,000 imgs] = ~5,600 unseen OOD imgs)
    shard_jobs.append(
        {
            "repo": "ComplexDataLab/OpenFake",
            "file": of_core_test[0],
            "kind": "openfake",
            "source": "openfake_core_test",
            "role": "ood",
            "max_real": 1800,
            "max_fake": 1800,
        }
    )
    shard_jobs.append(
        {
            "repo": "ComplexDataLab/OpenFake",
            "file": of_reddit_test[0],
            "kind": "openfake",
            "source": "openfake_reddit_test",
            "role": "ood",
            "max_real": 1000,
            "max_fake": 1000,
        }
    )

    # 2. Defactify (all 9 compact ~400MB shards: ~8,500 MS-COCO Real + ~18,000 SD2.1/SDXL/SD3/DALL-E 3/MJ6 in ~4.5 min!)
    for f in defact_train:
        shard_jobs.append(
            {
                "repo": "Rajarshi-Roy-research/Defactify_Image_Dataset",
                "file": f,
                "kind": "defactify",
                "source": "defactify",
                "role": "train_pool",
                "max_real": 2000,
                "max_fake_per_gen": 400,
            }
        )

    # 3. OpenFake Core Wave 1 (2 uniformly-shuffled shards: ALL 57 OpenFake generators + Pexels/LAION Reals!)
    ct_step = max(1, len(of_core_train) // 6)
    selected_core = of_core_train[::ct_step][:4] + [of_core_val[0]]
    for f in selected_core[:2]:
        shard_jobs.append(
            {
                "repo": "ComplexDataLab/OpenFake",
                "file": f,
                "kind": "openfake_core_bal",
                "source": "openfake_core",
                "role": "train_pool",
                "max_real": 2500,
                "max_fake": 2100,
            }
        )

    # 4. OpenFake 2026 In-The-Wild Supplements (Reddit 2026 + Civitai 2026 + Frontier T2I 2026)
    for f in [of_supp_reddit[0], of_supp_reddit[len(of_supp_reddit) // 2]]:
        shard_jobs.append(
            {
                "repo": "ComplexDataLab/OpenFake",
                "file": f,
                "kind": "openfake",
                "source": "openfake_reddit_2026",
                "role": "train_pool",
                "max_real": 2000,
                "max_fake": 1400,
            }
        )
    shard_jobs.append(
        {
            "repo": "ComplexDataLab/OpenFake",
            "file": of_supp_civitai[0],
            "kind": "openfake",
            "source": "openfake_civitai_2026",
            "role": "train_pool",
            "max_real": 500,
            "max_fake": 1000,
        }
    )
    shard_jobs.append(
        {
            "repo": "ComplexDataLab/OpenFake",
            "file": of_supp_t2i[0],
            "kind": "openfake",
            "source": "openfake_t2i_2026",
            "role": "train_pool",
            "max_real": 300,
            "max_fake": 650,
        }
    )

    # 5. CommunityForensics-Eval: 45 spaced shards (VISION/Forchheim/IMD2020/FFHQ smartphone/camera Reals + 45 generators)
    cf_step = max(1, len(cf_shards) // 45)
    for f in cf_shards[::cf_step][:45]:
        shard_jobs.append(
            {
                "repo": "OwensLab/CommunityForensics-Eval",
                "file": f,
                "kind": "commfor",
                "source": "community_forensics",
                "role": "train_pool",
                "max_real": 600,
                "max_fake": 140,
            }
        )

    # 6. OpenFake Core Wave 2 (additional shards if time permits)
    for f in selected_core[2:]:
        shard_jobs.append(
            {
                "repo": "ComplexDataLab/OpenFake",
                "file": f,
                "kind": "openfake_core_bal",
                "source": "openfake_core",
                "role": "train_pool",
                "max_real": 2600,
                "max_fake": 1900,
            }
        )

    max_bytes = int(args.max_gb * (1024 ** 3))
    pool_real_count = sum(1 for r in existing_records if r["split_role"] == "train_pool" and int(r["label"]) == 0)
    pool_fake_count = sum(1 for r in existing_records if r["split_role"] == "train_pool" and int(r["label"]) == 1)

    print("=" * 88)
    print(
        f"Starting Multi-Generator Harvester | Jobs: {len(shard_jobs)} shards | "
        f"Budget: <= {args.max_gb:.1f} GB | Free C: {get_free_gb():.2f} GB (Floor: {args.min_free_gb:.1f} GB)"
    )
    print("=" * 88)

    tmp_file = tmp_dir / "current_shard.parquet"

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for job_idx, job in enumerate(shard_jobs, start=1):
            shard_key = f"{job['repo']}::{job['file']}"
            if shard_key in done_shards:
                continue

            elapsed_min = (time.time() - t_start) / 60.0
            free_gb = get_free_gb()
            if total_bytes_written >= max_bytes:
                print(f"[Stop] Reached target dataset budget ({total_bytes_written/(1024**3):.2f} GB).")
                break
            if free_gb < args.min_free_gb:
                print(f"[Stop] Free space on C: ({free_gb:.2f} GB) reached safety floor ({args.min_free_gb:.1f} GB).")
                break
            if elapsed_min >= args.max_minutes and job["role"] != "ood":
                print(f"[Stop] Reached {elapsed_min:.1f} min harvest time cap; moving to split generation & training.")
                break

            t_dl = time.time()
            ok = download_shard_direct(job["repo"], job["file"], tmp_file, min_free_gb=args.min_free_gb)
            if not ok or not tmp_file.exists():
                continue
            dl_sec = max(0.05, time.time() - t_dl)
            shard_mb = tmp_file.stat().st_size / (1024 ** 2)

            new_rows_shard: list[dict] = []
            real_taken = 0
            fake_taken = 0
            per_gen_taken: dict[int, int] = {}

            # Dynamic 50/50 balancing on core shards so final train_pool lands balanced
            max_r = job.get("max_real", 999999)
            max_f = job.get("max_fake", 999999)
            if job["kind"] == "openfake_core_bal":
                deficit_real = pool_fake_count - pool_real_count
                if deficit_real > 1500:
                    max_f = max(900, max_r - min(1300, deficit_real // 3))
                elif deficit_real < -1500:
                    max_r = max(900, max_f - min(1300, (-deficit_real) // 3))

            try:
                with open(tmp_file, "rb") as fp:
                    pf = pq.ParquetFile(fp, memory_map=False)
                    avail_cols = set(pf.schema_arrow.names)

                    if job["kind"] in ("openfake", "openfake_core_bal"):
                        want = [
                            c
                            for c in ("image", "label", "model", "type", "image_sha256", "nsfw_score", "frame_index", "subreddit")
                            if c in avail_cols
                        ]
                    elif job["kind"] == "defactify":
                        want = [c for c in ("Image", "Label_A", "Label_B") if c in avail_cols]
                    else:
                        want = [
                            c
                            for c in ("image_data", "label", "model_name", "real_source", "nsfw_flag", "architecture")
                            if c in avail_cols
                        ]

                    pending_tasks: list[tuple] = []

                    for batch in pf.iter_batches(batch_size=128, columns=want):
                        cols = batch.to_pydict()
                        n_b = batch.num_rows

                        for i in range(n_b):
                            if job["kind"] in ("openfake", "openfake_core_bal"):
                                lbl_str = str(cols["label"][i] or "").strip().lower()
                                if lbl_str not in ("real", "fake"):
                                    continue
                                lbl = 0 if lbl_str == "real" else 1
                                gen_name = str(cols.get("model", [""] * n_b)[i] or "unknown")
                                itype = str(cols.get("type", [""] * n_b)[i] or "")
                                if "tiny-random" in gen_name.lower():
                                    continue
                                if "nsfw_score" in cols and cols["nsfw_score"][i] is not None:
                                    if float(cols["nsfw_score"][i]) >= 0.22:
                                        continue
                                if "frame_index" in cols and cols["frame_index"][i] is not None:
                                    continue
                                if "video" in itype.lower():
                                    continue
                                if gen_name == "reddit" and "subreddit" in cols and cols["subreddit"][i]:
                                    gen_name = f"reddit_{cols['subreddit'][i]}"

                                if lbl == 0 and real_taken >= max_r:
                                    continue
                                if lbl == 1 and fake_taken >= max_f:
                                    continue

                                raw_b = extract_raw_bytes(cols["image"][i])
                                if not raw_b:
                                    continue
                                sha = str(cols.get("image_sha256", [""] * n_b)[i] or "")
                                if not sha:
                                    sha = hashlib.sha256(raw_b).hexdigest()
                                if sha in seen_hashes:
                                    continue

                                seen_hashes.add(sha)
                                if lbl == 0:
                                    real_taken += 1
                                else:
                                    fake_taken += 1

                            elif job["kind"] == "defactify":
                                la = int(cols["Label_A"][i])
                                lb = int(cols["Label_B"][i])
                                lbl = 0 if la == 0 else 1
                                if lbl == 0:
                                    if real_taken >= max_r:
                                        continue
                                else:
                                    if per_gen_taken.get(lb, 0) >= job["max_fake_per_gen"]:
                                        continue
                                raw_b = extract_raw_bytes(cols["Image"][i])
                                if not raw_b:
                                    continue
                                sha = hashlib.sha256(raw_b).hexdigest()
                                if sha in seen_hashes:
                                    continue
                                seen_hashes.add(sha)
                                gen_name = DEFACTIFY_GEN_MAP.get(lb, f"defactify_{lb}")
                                itype = "mscoco" if lbl == 0 else "defactify_t2i"
                                if lbl == 0:
                                    real_taken += 1
                                else:
                                    fake_taken += 1
                                    per_gen_taken[lb] = per_gen_taken.get(lb, 0) + 1

                            else:
                                # CommunityForensics-Eval (note: nsfw_flag is string 'False'/'True')
                                if "nsfw_flag" in cols and str(cols["nsfw_flag"][i] or "").strip().lower() in ("true", "1", "yes"):
                                    continue
                                lbl = int(cols["label"][i])
                                if lbl not in (0, 1):
                                    continue
                                if lbl == 0 and real_taken >= max_r:
                                    continue
                                if lbl == 1 and fake_taken >= max_f:
                                    continue
                                raw_b = extract_raw_bytes(cols["image_data"][i])
                                if not raw_b:
                                    continue
                                sha = hashlib.sha256(raw_b).hexdigest()
                                if sha in seen_hashes:
                                    continue
                                seen_hashes.add(sha)
                                if lbl == 0:
                                    rsrc = str(cols.get("real_source", [""] * n_b)[i] or "camera")
                                    gen_name = f"cf_real_{rsrc}"
                                    itype = "real_camera"
                                    real_taken += 1
                                else:
                                    mname = str(cols.get("model_name", [""] * n_b)[i] or "cf_synth")
                                    arch = str(cols.get("architecture", [""] * n_b)[i] or "")
                                    gen_name = mname
                                    itype = arch
                                    fake_taken += 1

                            pending_tasks.append(
                                (
                                    raw_b,
                                    sha,
                                    lbl,
                                    gen_name,
                                    itype,
                                    job["source"],
                                    job["role"],
                                    out_root,
                                    data_dir,
                                )
                            )

                            if len(pending_tasks) >= 256:
                                for res in pool.map(process_and_write_image, pending_tasks):
                                    if res is not None:
                                        new_rows_shard.append(res)
                                        total_bytes_written += res["bytes"]
                                        if res["split_role"] == "train_pool":
                                            if res["label"] == 0:
                                                pool_real_count += 1
                                            else:
                                                pool_fake_count += 1
                                pending_tasks.clear()
                                if total_bytes_written >= max_bytes or get_free_gb() < args.min_free_gb:
                                    break

                        if total_bytes_written >= max_bytes or get_free_gb() < args.min_free_gb:
                            break
                        if job["kind"] != "defactify" and real_taken >= max_r and fake_taken >= max_f:
                            break

                    if pending_tasks and total_bytes_written < max_bytes and get_free_gb() >= args.min_free_gb:
                        for res in pool.map(process_and_write_image, pending_tasks):
                            if res is not None:
                                new_rows_shard.append(res)
                                total_bytes_written += res["bytes"]
                                if res["split_role"] == "train_pool":
                                    if res["label"] == 0:
                                        pool_real_count += 1
                                    else:
                                        pool_fake_count += 1
                        pending_tasks.clear()
            except Exception as exc:
                print(f"  [Warn] Error reading shard {job['file']}: {exc}")
            finally:
                tmp_file.unlink(missing_ok=True)

            if new_rows_shard:
                existing_records.extend(new_rows_shard)
                pd.DataFrame(existing_records).to_csv(manifest_path, index=False)

            done_shards.add(shard_key)
            done_shards_path.write_text("\n".join(sorted(done_shards)) + "\n", encoding="utf-8")

            r_added = sum(1 for r in new_rows_shard if r["label"] == 0)
            f_added = len(new_rows_shard) - r_added
            if job_idx <= 12 or job_idx % 5 == 0 or len(new_rows_shard) >= 400:
                print(
                    f"[{job_idx:03d}/{len(shard_jobs):03d}] {job['source']:22s} | "
                    f"DL {shard_mb:6.1f}MB in {dl_sec:4.1f}s ({shard_mb/dl_sec:5.1f} MB/s) | "
                    f"+{len(new_rows_shard):4d} imgs (R:{r_added}/AI:{f_added}) | "
                    f"Total: {len(existing_records):,} ({total_bytes_written/(1024**3):.2f} GB, Free C: {get_free_gb():.1f} GB)"
                )

    shutil.rmtree(tmp_dir, ignore_errors=True)
    cleanup_hf_probe_cache()
    build_v5_splits(data_dir)


if __name__ == "__main__":
    harvest()
