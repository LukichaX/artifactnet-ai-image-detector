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
from tqdm import tqdm

from src.dataset import create_dataloaders, prepare_splits
from src.forensic_council import ForensicCouncil
from src.model import ArtifactNetV4
from src.utils import compute_metrics, evaluate_perturbation_suite, seed_everything


def resolve_v4_checkpoint(ckpt_dir: Path) -> Path:
    v4_path = ckpt_dir / "best_model_v4.pth"
    if v4_path.exists():
        return v4_path
    generic_path = ckpt_dir / "best_model.pth"
    if generic_path.exists():
        meta = torch.load(generic_path, map_location="cpu", weights_only=False)
        if isinstance(meta, dict) and meta.get("architecture") == "ArtifactNetV4":
            return generic_path
    raise FileNotFoundError(
        "No trained ArtifactNet v4 checkpoint found yet.\n"
        "Run `python src/train.py` first to train ArtifactNet v4 from scratch."
    )


def evaluate() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ArtifactNet v4 on Hold-Out & Stress Benchmarks")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--stress-samples", type=int, default=1200, help="Test samples per OOD perturbation benchmark")
    parser.add_argument("--skip-submission", action="store_true", help="Skip 5,540-image test_data_v2 TTA submission")
    args = parser.parse_args()

    seed_everything(42)
    data_dir = PROJECT_ROOT / "data"
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    assets_dir = PROJECT_ROOT / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = resolve_v4_checkpoint(ckpt_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_type = "cuda" if device.type == "cuda" else "cpu"
    use_amp = device.type == "cuda"

    model = ArtifactNetV4(dropout=0.0).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    print("=" * 88)
    print(
        f"Evaluating ArtifactNet v4 ({ckpt_path.name}, Epoch {ckpt.get('epoch', '?')}) | "
        f"Device: {torch.cuda.get_device_name(0) if use_amp else 'CPU'}"
    )
    print("=" * 88)

    _, _, test_df = prepare_splits(data_dir)
    _, _, test_loader = create_dataloaders(
        data_dir=data_dir,
        img_size=256,
        batch_size=args.batch_size,
        num_workers=args.workers,
        return_global=True,
        paired_train=False,
    )

    all_targets, all_probs = [], []
    with torch.no_grad():
        for x_global, x_rich, x_poor, targets in tqdm(test_loader, desc="1/7 Clean Hold-Out Test"):
            x_global = x_global.to(device, non_blocking=True)
            x_rich = x_rich.to(device, non_blocking=True)
            x_poor = x_poor.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_amp):
                logits, det = model(x_rich, x_poor, x_global=x_global, return_details=True)
                l_f32 = logits.to(torch.float32)
                if det["stat_calibrated"]:
                    l_f32 = 0.90 * l_f32 + 0.10 * det["stat_logit"].to(torch.float32)
                probs = torch.sigmoid(l_f32).squeeze(1)

            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_targets)
    y_probs = np.array(all_probs)
    clean_m = compute_metrics(y_true, y_probs)

    print("\n[Hold-Out Test Split Results (7,996 Untouched Images)]")
    print(f"  • F1-Score        : {clean_m['f1']*100:.2f}%")
    print(f"  • Accuracy        : {clean_m['accuracy']*100:.2f}%")
    print(f"  • Precision (AI)  : {clean_m['precision']*100:.2f}%")
    print(f"  • Recall (AI)     : {clean_m['recall']*100:.2f}%")
    print(f"  • Specificity (R) : {clean_m['specificity']*100:.2f}%")
    print(f"  • ROC-AUC         : {clean_m['roc_auc']*100:.2f}%")

    stress_modes = (
        "upscale_1_8x",
        "downscale_0_68x_png",
        "jpeg_q96_444",
        "phone_isp",
        "fft_hf_knockout",
    )
    print(f"\nRunning 5 Out-of-Domain & Anti-Shortcut Stress Tests ({args.stress_samples} test images each)...")
    stress_res = evaluate_perturbation_suite(
        model,
        test_df,
        data_dir,
        device,
        modes=stress_modes,
        samples_per_mode=args.stress_samples,
        batch_size=args.batch_size,
        num_workers=max(2, args.workers // 2),
    )

    mode_labels = {
        "upscale_1_8x": "1.8x Bicubic Upscale",
        "downscale_0_68x_png": "0.68x Area + PNG (No Q75 Grid)",
        "jpeg_q96_444": "JPEG Q96 (4:4:4) + Grid Shift",
        "phone_isp": "Smartphone ISP Denoise+Sharpen",
        "fft_hf_knockout": "FFT High-Freq Ring Knock-Out",
    }

    print("\n[Cross-Domain & Shortcut-Knockout Stress Benchmark]")
    print(f"{'Condition':34s} | {'F1':>7s} | {'Acc':>7s} | {'Real Spec':>9s} | {'AI Recall':>9s} | {'ROC-AUC':>8s}")
    print("-" * 88)
    print(
        f"{'Clean Hold-Out (Baseline)':34s} | {clean_m['f1']*100:6.2f}% | {clean_m['accuracy']*100:6.2f}% | "
        f"{clean_m['specificity']*100:8.2f}% | {clean_m['recall']*100:8.2f}% | {clean_m['roc_auc']*100:7.2f}%"
    )
    for m_key in stress_modes:
        sm = stress_res[m_key]
        print(
            f"{mode_labels[m_key]:34s} | {sm['f1']*100:6.2f}% | {sm['accuracy']*100:6.2f}% | "
            f"{sm['specificity']*100:8.2f}% | {sm['recall']*100:8.2f}% | {sm['roc_auc']*100:7.2f}%"
        )

    # --- Save 3-Panel Evaluation Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))

    cm = confusion_matrix(y_true, (y_probs >= 0.5).astype(int))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real (0)", "AI (1)"],
        yticklabels=["Real (0)", "AI (1)"],
        ax=axes[0],
    )
    axes[0].set_title(f"Hold-Out Confusion Matrix (F1={clean_m['f1']*100:.2f}%)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    axes[1].plot(fpr, tpr, label=f"Hold-Out AUC = {clean_m['roc_auc']*100:.2f}%", linewidth=2.2)
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.35)
    axes[1].set_title("Hold-Out ROC Curve")
    axes[1].set_xlabel("False Positive Rate (1 - Real Specificity)")
    axes[1].set_ylabel("True Positive Rate (AI Recall)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="lower right")

    bar_names = ["Clean"] + ["1.8x Up", "0.68x PNG", "Q96 4:4:4", "Phone ISP", "FFT Knock"]
    spec_vals = [clean_m["specificity"] * 100.0] + [stress_res[k]["specificity"] * 100.0 for k in stress_modes]
    rec_vals = [clean_m["recall"] * 100.0] + [stress_res[k]["recall"] * 100.0 for k in stress_modes]
    x_pos = np.arange(len(bar_names))
    width = 0.36
    axes[2].bar(x_pos - width / 2, spec_vals, width, label="Real Specificity (%)", color="#2b6cb0")
    axes[2].bar(x_pos + width / 2, rec_vals, width, label="AI Recall (%)", color="#dd6b20")
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(bar_names, rotation=22, ha="right")
    axes[2].set_ylim(0, 105)
    axes[2].set_title("Cross-Domain Robustness Stress Suite")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(loc="lower right")

    plt.tight_layout()
    out_fig = assets_dir / "test_evaluation_v4.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"\nSaved evaluation report chart -> {out_fig}")

    # --- 7/7: Unlabeled test_data_v2 Multi-Scale 4-View TTA Prediction ---
    test_csv = data_dir / "test.csv"
    if not args.skip_submission and test_csv.exists():
        council = ForensicCouncil(model=model, device=device)
        sub_df = pd.read_csv(test_csv)
        probs_out, preds_out = [], []

        for _, row in tqdm(sub_df.iterrows(), total=len(sub_df), desc="7/7 Predicting test_data_v2 (4-View TTA)"):
            img_path = data_dir / row["id"]
            if not img_path.exists():
                img_path = data_dir / "test_data_v2" / Path(row["id"]).name
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                probs_out.append(0.0)
                preds_out.append(0)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            prob = council.predict_prob(rgb, use_tta=True)
            probs_out.append(prob)
            preds_out.append(1 if prob >= 0.5 else 0)

        sub_df["label"] = preds_out
        sub_v4_path = PROJECT_ROOT / "submission_v4.csv"
        sub_main_path = PROJECT_ROOT / "submission.csv"
        sub_df.to_csv(sub_v4_path, index=False)
        sub_df.to_csv(sub_main_path, index=False)

        prob_df = sub_df.copy()
        prob_df["ai_probability"] = np.round(probs_out, 6)
        prob_df.to_csv(PROJECT_ROOT / "submission_probs_v4.csv", index=False)

        n_ai = int(np.sum(preds_out))
        n_real = len(preds_out) - n_ai
        print(
            f"[Submission Ready] {sub_v4_path.name} & {sub_main_path.name} | "
            f"Real: {n_real} ({n_real/len(preds_out)*100:.1f}%) | AI: {n_ai} ({n_ai/len(preds_out)*100:.1f}%)"
        )


if __name__ == "__main__":
    evaluate()
