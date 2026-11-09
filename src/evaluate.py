import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.dataset import SingleForensicDataset, ensure_rgb_uint8, prepare_splits, prepare_splits_v5, strip_borders_and_canvas
from src.forensic_council import ForensicCouncil
from src.model import ArtifactNetV5
from src.utils import compute_metrics, evaluate_perturbation_suite, seed_everything


def resolve_v5_checkpoint(ckpt_dir: Path) -> Path:
    v5_path = ckpt_dir / "best_model_v5.pth"
    if v5_path.exists():
        return v5_path
    generic_path = ckpt_dir / "best_model.pth"
    if generic_path.exists():
        meta = torch.load(generic_path, map_location="cpu", weights_only=False)
        if isinstance(meta, dict) and meta.get("architecture") == "ArtifactNetV5":
            return generic_path
    raise FileNotFoundError(
        "No trained ArtifactNet v5 checkpoint found yet.\n"
        "Run `python src/train.py` first to train ArtifactNet v5 from scratch."
    )


class _TTASubmissionDataset(Dataset):
    def __init__(self, sub_df: pd.DataFrame, data_dir: Path) -> None:
        self.ids = sub_df["id"].astype(str).tolist()
        self.data_dir = data_dir

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        rel = self.ids[idx]
        p = self.data_dir / rel
        if not p.exists():
            p = self.data_dir / "test_data_v2" / Path(rel).name
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            return torch.zeros((4, 3, 256, 256), dtype=torch.uint8), torch.zeros((4, 3, 256, 256), dtype=torch.uint8), torch.zeros((4, 3, 256, 256), dtype=torch.uint8), False
        rgb = ensure_rgb_uint8(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        clean_img, _, _ = strip_borders_and_canvas(rgb)
        xg, xr, xp = ForensicCouncil._build_views(clean_img, img_size=256, use_tta=True)
        return (
            torch.from_numpy(np.ascontiguousarray(xg.transpose(0, 3, 1, 2))),
            torch.from_numpy(np.ascontiguousarray(xr.transpose(0, 3, 1, 2))),
            torch.from_numpy(np.ascontiguousarray(xp.transpose(0, 3, 1, 2))),
            True,
        )


@torch.no_grad()
def _predict_df(
    model: ArtifactNetV5,
    df: pd.DataFrame,
    data_dir: Path,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
    desc: str = "Eval",
) -> tuple[np.ndarray, np.ndarray]:
    ds = SingleForensicDataset(df, data_dir, img_size=256, perturbation="clean", return_global=True)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    dev_type = "cuda" if device.type == "cuda" else "cpu"
    use_amp = device.type == "cuda"
    is_tty = sys.stdout.isatty()

    all_t, all_p = [], []
    for xg, xr, xp, targets in tqdm(loader, desc=desc, disable=not is_tty):
        xg = xg.to(device, non_blocking=True)
        xr = xr.to(device, non_blocking=True)
        xp = xp.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_amp):
            logits, det = model(x_rich=xr, x_poor=xp, x_global=xg, return_details=True)
            l_f32 = logits.to(torch.float32)
            if det["stat_calibrated"]:
                l_f32 = 0.90 * l_f32 + 0.10 * det["stat_logit"].to(torch.float32)
            probs = torch.sigmoid(l_f32).squeeze(1)
        all_t.extend(targets.numpy())
        all_p.extend(probs.cpu().numpy())

    return np.array(all_t), np.array(all_p)


def evaluate() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ArtifactNet v5 on Unseen-Generator & Internet-Laundered Benchmarks")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--stress-samples", type=int, default=1400, help="Unseen OOD samples per internet-laundering stress mode")
    parser.add_argument("--skip-submission", action="store_true", help="Skip 5,540-image test_data_v2 4-view TTA submission")
    args = parser.parse_args()

    seed_everything(42)
    data_dir = PROJECT_ROOT / "data"
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    assets_dir = PROJECT_ROOT / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = resolve_v5_checkpoint(ckpt_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_type = "cuda" if device.type == "cuda" else "cpu"
    use_amp = device.type == "cuda"
    is_tty = sys.stdout.isatty()

    model = ArtifactNetV5(dropout=0.0).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    print("=" * 94)
    print(
        f"Evaluating ArtifactNet v5 ({ckpt_path.name}, Epoch {ckpt.get('epoch', '?')}) | "
        f"Device: {torch.cuda.get_device_name(0) if use_amp else 'CPU'}"
    )
    print("=" * 94)

    _, val_id_df, _, test_ood_df = prepare_splits_v5(data_dir)

    # 1. Strictly Unseen-Generator & Reddit In-The-Wild Hold-Out Test (`test_ood_split.csv`)
    y_ood, p_ood = _predict_df(
        model, test_ood_df, data_dir, device, batch_size=args.batch_size, num_workers=args.workers, desc="1/4 Unseen-Gen OOD Test"
    )
    ood_m = compute_metrics(y_ood, p_ood)

    print(f"\n[1. Strictly Unseen-Generator & In-The-Wild OOD Test ({len(test_ood_df):,} Images)]")
    print(f"  • F1-Score        : {ood_m['f1']*100:.2f}%")
    print(f"  • Accuracy        : {ood_m['accuracy']*100:.2f}%")
    print(f"  • Precision (AI)  : {ood_m['precision']*100:.2f}%")
    print(f"  • Recall (AI)     : {ood_m['recall']*100:.2f}%")
    print(f"  • Specificity (R) : {ood_m['specificity']*100:.2f}%")
    print(f"  • ROC-AUC         : {ood_m['roc_auc']*100:.2f}%")

    # Per-generator / per-camera breakdown on unseen OOD test!
    if "generator" in test_ood_df.columns:
        gen_eval = test_ood_df.copy()
        gen_eval["prob"] = p_ood
        gen_eval["pred"] = (p_ood >= 0.5).astype(int)
        gen_eval["correct"] = (gen_eval["pred"] == gen_eval["label"].astype(int)).astype(float)
        print("\n[Per-Generator / Source Breakdown on Unseen OOD Test]")
        print(f"  {'Generator / Camera Source':34s} | {'Class':>5s} | {'Count':>6s} | {'Accuracy':>9s} | {'Mean AI Prob':>12s}")
        print("  " + "-" * 78)
        grp_stats = (
            gen_eval.groupby(["label", "generator"], as_index=False)
            .agg(count=("correct", "size"), acc=("correct", "mean"), mean_p=("prob", "mean"))
            .sort_values(["label", "count"], ascending=[True, False])
        )
        for _, r in grp_stats.iterrows():
            if int(r["count"]) < 15:
                continue
            cls_tag = "Real" if int(r["label"]) == 0 else "AI"
            print(
                f"  {str(r['generator'])[:34]:34s} | {cls_tag:>5s} | {int(r['count']):6d} | "
                f"{float(r['acc'])*100:8.2f}% | {float(r['mean_p'])*100:11.2f}%"
            )

    # 2. Multi-Generator In-Distribution Hold-Out + Legacy 7,996 Test Split
    y_id, p_id = _predict_df(
        model, val_id_df, data_dir, device, batch_size=args.batch_size, num_workers=args.workers, desc="2/4 Multi-Gen ID Val"
    )
    id_m = compute_metrics(y_id, p_id)
    print(
        f"\n[2. Multi-Generator In-Distribution Split ({len(val_id_df):,} Images)] "
        f"F1: {id_m['f1']*100:.2f}% | Acc: {id_m['accuracy']*100:.2f}% | "
        f"Spec: {id_m['specificity']*100:.2f}% | Rec: {id_m['recall']*100:.2f}% | AUC: {id_m['roc_auc']*100:.2f}%"
    )

    leg_m = None
    y_leg, p_leg = None, None
    if (data_dir / "splits" / "test_split.csv").exists():
        _, _, leg_test_df = prepare_splits(data_dir)
        y_leg, p_leg = _predict_df(
            model, leg_test_df, data_dir, device, batch_size=args.batch_size, num_workers=args.workers, desc="3/4 Legacy Hold-Out"
        )
        leg_m = compute_metrics(y_leg, p_leg)
        print(
            f"[3. Original Dataset Hold-Out Split ({len(leg_test_df):,} Images)]     "
            f"F1: {leg_m['f1']*100:.2f}% | Acc: {leg_m['accuracy']*100:.2f}% | "
            f"Spec: {leg_m['specificity']*100:.2f}% | Rec: {leg_m['recall']*100:.2f}% | AUC: {leg_m['roc_auc']*100:.2f}%"
        )

    # 3. 6-Condition Internet Laundering & Shortcut-Knockout Stress Benchmark on Unseen OOD!
    stress_modes = (
        "whatsapp_1hop",
        "social_2hop_chain",
        "webp_cdn_q75",
        "screenshot_reencode",
        "phone_isp",
        "fft_hf_knockout",
    )
    print(f"\nRunning 6 Internet-Laundering & Anti-Shortcut Stress Tests on Unseen OOD ({args.stress_samples} images each)...")
    stress_res = evaluate_perturbation_suite(
        model,
        test_ood_df,
        data_dir,
        device,
        modes=stress_modes,
        samples_per_mode=args.stress_samples,
        batch_size=args.batch_size,
        num_workers=max(2, args.workers // 2),
    )

    mode_labels = {
        "whatsapp_1hop": "WhatsApp 1-Hop (1280px + Q74 4:2:0)",
        "social_2hop_chain": "2-Hop Upload->Download->Forward Chain",
        "webp_cdn_q75": "WebP Q75 CDN Re-Encode",
        "screenshot_reencode": "Compressed Web -> Resized PNG Save",
        "phone_isp": "Smartphone ISP Denoise+Sharpen",
        "fft_hf_knockout": "FFT High-Freq Ring Knock-Out",
    }

    print("\n[Unseen-Generator + Internet-Laundering Stress Benchmark]")
    print(f"{'Delivery / Laundering Condition':38s} | {'F1':>7s} | {'Acc':>7s} | {'Real Spec':>9s} | {'AI Recall':>9s} | {'ROC-AUC':>8s}")
    print("-" * 92)
    print(
        f"{'Clean Unseen-Gen OOD (Baseline)':38s} | {ood_m['f1']*100:6.2f}% | {ood_m['accuracy']*100:6.2f}% | "
        f"{ood_m['specificity']*100:8.2f}% | {ood_m['recall']*100:8.2f}% | {ood_m['roc_auc']*100:7.2f}%"
    )
    for m_key in stress_modes:
        sm = stress_res[m_key]
        print(
            f"{mode_labels[m_key]:38s} | {sm['f1']*100:6.2f}% | {sm['accuracy']*100:6.2f}% | "
            f"{sm['specificity']*100:8.2f}% | {sm['recall']*100:8.2f}% | {sm['roc_auc']*100:7.2f}%"
        )

    # --- Save 3-Panel v5 Evaluation Report Chart ---
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.5))

    cm = confusion_matrix(y_ood, (p_ood >= 0.5).astype(int))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real (0)", "AI (1)"],
        yticklabels=["Real (0)", "AI (1)"],
        ax=axes[0],
    )
    axes[0].set_title(f"Unseen-Generator OOD Confusion Matrix (F1={ood_m['f1']*100:.2f}%)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    fpr_ood, tpr_ood, _ = roc_curve(y_ood, p_ood)
    axes[1].plot(fpr_ood, tpr_ood, label=f"Unseen-Gen OOD AUC = {ood_m['roc_auc']*100:.2f}%", linewidth=2.3)
    fpr_id, tpr_id, _ = roc_curve(y_id, p_id)
    axes[1].plot(fpr_id, tpr_id, label=f"Multi-Gen ID AUC = {id_m['roc_auc']*100:.2f}%", linewidth=2.0, linestyle="--")
    if leg_m is not None and y_leg is not None:
        fpr_l, tpr_l, _ = roc_curve(y_leg, p_leg)
        axes[1].plot(fpr_l, tpr_l, label=f"Legacy Hold-Out AUC = {leg_m['roc_auc']*100:.2f}%", linewidth=1.8, linestyle=":")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.30)
    axes[1].set_title("ROC Curves Across Seen & Unseen Generators")
    axes[1].set_xlabel("False Positive Rate (1 - Real Specificity)")
    axes[1].set_ylabel("True Positive Rate (AI Recall)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="lower right")

    bar_names = ["Clean OOD", "WhatsApp", "2-Hop Web", "WebP CDN", "Screen PNG", "Phone ISP", "FFT Knock"]
    spec_vals = [ood_m["specificity"] * 100.0] + [stress_res[k]["specificity"] * 100.0 for k in stress_modes]
    rec_vals = [ood_m["recall"] * 100.0] + [stress_res[k]["recall"] * 100.0 for k in stress_modes]
    x_pos = np.arange(len(bar_names))
    width = 0.36
    axes[2].bar(x_pos - width / 2, spec_vals, width, label="Real Specificity (%)", color="#2b6cb0")
    axes[2].bar(x_pos + width / 2, rec_vals, width, label="AI Recall (%)", color="#dd6b20")
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(bar_names, rotation=24, ha="right")
    axes[2].set_ylim(0, 105)
    axes[2].set_title("Unseen-Generator Internet Laundering Robustness")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(loc="lower right")

    plt.tight_layout()
    out_fig = assets_dir / "test_evaluation_v5.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"\nSaved evaluation report chart -> {out_fig}")

    # --- Batched 4-View TTA Prediction on Unlabeled test_data_v2 (5,540 Images) ---
    test_csv = data_dir / "test.csv"
    if not args.skip_submission and test_csv.exists():
        sub_df = pd.read_csv(test_csv)
        sub_ds = _TTASubmissionDataset(sub_df, data_dir)
        sub_loader = DataLoader(
            sub_ds,
            batch_size=16,
            shuffle=False,
            num_workers=max(2, args.workers // 2),
            pin_memory=use_amp,
        )
        probs_out, preds_out = [], []
        with torch.no_grad():
            for xg_b, xr_b, xp_b, valid_b in tqdm(sub_loader, desc="4/4 Predicting test_data_v2 (4-View TTA)", disable=not is_tty):
                bsz = xg_b.shape[0]
                xg_flat = xg_b.view(bsz * 4, 3, 256, 256).to(device, non_blocking=True)
                xr_flat = xr_b.view(bsz * 4, 3, 256, 256).to(device, non_blocking=True)
                xp_flat = xp_b.view(bsz * 4, 3, 256, 256).to(device, non_blocking=True)

                with torch.amp.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_amp):
                    f_logit, det = model(xr_flat, xp_flat, x_global=xg_flat, return_details=True)

                l_mean = f_logit.to(torch.float32).view(bsz, 4).mean(dim=1)
                if det["stat_calibrated"]:
                    s_mean = det["stat_logit"].to(torch.float32).view(bsz, 4).mean(dim=1)
                    l_mean = 0.90 * l_mean + 0.10 * s_mean

                p_batch = torch.sigmoid(l_mean).cpu().numpy()
                for p_val, ok_flag in zip(p_batch, valid_b.numpy()):
                    prob = float(p_val) if bool(ok_flag) else 0.0
                    probs_out.append(prob)
                    preds_out.append(1 if prob >= 0.5 else 0)

        sub_df["label"] = preds_out
        sub_v5_path = PROJECT_ROOT / "submission_v5.csv"
        sub_main_path = PROJECT_ROOT / "submission.csv"
        sub_df.to_csv(sub_v5_path, index=False)
        sub_df.to_csv(sub_main_path, index=False)

        prob_df = sub_df.copy()
        prob_df["ai_probability"] = np.round(probs_out, 6)
        prob_df.to_csv(PROJECT_ROOT / "submission_probs_v5.csv", index=False)

        n_ai = int(np.sum(preds_out))
        n_real = len(preds_out) - n_ai
        print(
            f"[Submission Ready] {sub_v5_path.name} & {sub_main_path.name} | "
            f"Real: {n_real} ({n_real/len(preds_out)*100:.1f}%) | AI: {n_ai} ({n_ai/len(preds_out)*100:.1f}%)"
        )


if __name__ == "__main__":
    evaluate()
