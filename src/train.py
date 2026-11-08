import argparse
import math
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.dataset import create_dataloaders, prepare_splits
from src.model import ArtifactNetV4
from src.utils import (
    STREAM_KEYS,
    ArtifactNetV4Loss,
    ModelEMA,
    compute_metrics,
    evaluate_perturbation_suite,
    fit_post_training_calibration,
    seed_everything,
)


def backup_legacy_checkpoints(ckpt_dir: Path) -> None:
    """Safely archives any pre-v4 checkpoints into `checkpoints/archive_v2/` before saving v4 weights."""
    best_generic = ckpt_dir / "best_model.pth"
    if not best_generic.exists():
        return
    try:
        meta = torch.load(best_generic, map_location="cpu", weights_only=False)
        arch = meta.get("architecture", "") if isinstance(meta, dict) else ""
    except Exception:
        arch = ""

    if arch == "ArtifactNetV4":
        return

    archive_dir = ckpt_dir / "archive_v2"
    archive_dir.mkdir(parents=True, exist_ok=True)
    legacy_names = [
        "best_model.pth",
        "last_model.pth",
        "best_model_v2.pth",
        "last_model_v2.pth",
        "training_history.csv",
        "training_history_v2.csv",
    ]
    copied = []
    for name in legacy_names:
        src = ckpt_dir / name
        dst = archive_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied.append(name)
    if copied:
        print(f"[Backup] Archived legacy checkpoints to {archive_dir}: {', '.join(copied)}")


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    total_steps: int,
    warmup_steps: int,
):
    """Decouples weight decay: applies WD only to 2D+ Conv/Linear weights, never to norms/biases/prototypes."""
    decay_params, no_decay_params = [], []
    no_decay_names = {"pos_embed", "manifold_tau", "manifold_log_scale"}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or any(nd in name for nd in no_decay_names) or "norm" in name or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.999),
    )

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = 1e-6 / max(lr, 1e-8)
        return min_ratio + (1.0 - min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def save_training_curves(history_df: pd.DataFrame, out_path: Path) -> None:
    if len(history_df) == 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ep = history_df["epoch"]
    axes[0].plot(ep, history_df["train_loss"], marker="o", label="Train Loss")
    axes[0].plot(ep, history_df["val_loss"], marker="s", label="Val Loss")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(ep, history_df["val_f1"] * 100.0, marker="o", label="Clean Val F1 (%)")
    axes[1].plot(ep, history_df["perturbed_f1"] * 100.0, marker="s", label="Perturbed OOD F1 (%)")
    axes[1].plot(ep, history_df["gen_score"] * 100.0, marker="^", linewidth=2.2, label="Harmonic Gen Score (%)")
    axes[1].set_title("In-Domain vs Scale/Codec/ISP Generalization")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score (%)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(ep, history_df["perturbed_specificity"] * 100.0, marker="o", label="Perturbed Real Specificity (%)")
    axes[2].plot(ep, history_df["perturbed_recall"] * 100.0, marker="s", label="Perturbed AI Recall (%)")
    axes[2].plot(ep, history_df["val_roc_auc"] * 100.0, marker="^", label="Clean Val ROC-AUC (%)")
    axes[2].set_title("Specificity & Recall Under Domain Shifts")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Percentage (%)")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ArtifactNet v4 From Scratch")
    parser.add_argument("--epochs", type=int, default=18, help="Total training epochs (default: 18, ~2.2h on RTX 5060 Ti)")
    parser.add_argument("--batch-size", type=int, default=32, help="Micro-batch images per GPU pass (must be even for matched pairs; default: 32)")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps (default: 2 -> effective batch size 64)")
    parser.add_argument("--lr", type=float, default=8e-4, help="Peak learning rate for AdamW")
    parser.add_argument("--weight-decay", type=float, default=2e-2, help="Weight decay on Conv/Linear matrices")
    parser.add_argument("--workers", type=int, default=6, help="DataLoader CPU worker processes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints/last_model_v4.pth if available")
    parser.add_argument("--calibrate-only", action="store_true", help="Run multi-condition calibration on existing best_model_v4.pth")
    return parser.parse_args()


def train() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data_dir = PROJECT_ROOT / "data"
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    assets_dir = PROJECT_ROOT / "assets"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    backup_legacy_checkpoints(ckpt_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_type = "cuda" if device.type == "cuda" else "cpu"
    use_amp = device.type == "cuda"
    gpu_label = torch.cuda.get_device_name(0) if use_amp else "CPU"

    _, val_df, _ = prepare_splits(data_dir)
    best_v4_path = ckpt_dir / "best_model_v4.pth"
    best_generic_path = ckpt_dir / "best_model.pth"
    last_v4_path = ckpt_dir / "last_model_v4.pth"
    history_csv = ckpt_dir / "training_history_v4.csv"

    if args.calibrate_only:
        target_ckpt = best_v4_path if best_v4_path.exists() else best_generic_path
        if not target_ckpt.exists():
            raise FileNotFoundError(f"No trained ArtifactNetV4 checkpoint found at {target_ckpt}")
        ckpt = torch.load(target_ckpt, map_location=device, weights_only=False)
        model = ArtifactNetV4(dropout=0.0).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"[Calibrate] Fitting multi-condition calibration on {target_ckpt.name}...")
        cal_info = fit_post_training_calibration(
            model, val_df, data_dir, device, samples_per_cond=600, batch_size=args.batch_size, num_workers=max(2, args.workers // 2)
        )
        ckpt["model_state_dict"] = model.state_dict()
        ckpt["calibration"] = cal_info
        torch.save(ckpt, best_v4_path)
        shutil.copy2(best_v4_path, best_generic_path)
        print(f"[Calibrate] Saved calibrated weights -> {best_v4_path.name} & {best_generic_path.name}")
        print(f"            {cal_info}")
        return

    batch_size = max(4, (args.batch_size // 2) * 2)
    grad_accum = max(1, args.grad_accum)

    train_loader, val_loader, _ = create_dataloaders(
        data_dir=data_dir,
        img_size=256,
        batch_size=batch_size,
        num_workers=args.workers,
        return_global=True,
        paired_train=True,
    )

    model = ArtifactNetV4(dropout=0.25, expert_drop_prob=0.15, spectral_drop_prob=0.50).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    steps_per_epoch = max(1, len(train_loader) // grad_accum)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = min(total_steps // 5, 2 * steps_per_epoch)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model, lr=args.lr, weight_decay=args.weight_decay, total_steps=total_steps, warmup_steps=warmup_steps
    )
    ema = ModelEMA(model, decay=0.997)
    criterion = ArtifactNetV4Loss(smoothing=0.03)

    start_epoch = 1
    best_gen_score = 0.0
    best_clean_f1 = 0.0
    history: list[dict] = []

    if args.resume and last_v4_path.exists():
        ckpt = torch.load(last_v4_path, map_location=device, weights_only=False)
        if ckpt.get("architecture") == "ArtifactNetV4":
            model.load_state_dict(ckpt["model_state_dict"])
            if "ema_state_dict" in ckpt:
                ema.ema.load_state_dict(ckpt["ema_state_dict"])
            else:
                ema.ema.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_gen_score = float(ckpt.get("best_gen_score", 0.0))
            best_clean_f1 = float(ckpt.get("best_clean_f1", 0.0))
            if history_csv.exists():
                history = pd.read_csv(history_csv).to_dict(orient="records")
            print(
                f"[Resume] Continuing ArtifactNet v4 from Epoch {start_epoch}/{args.epochs} "
                f"(Best Gen Score: {best_gen_score*100:.2f}%)"
            )

    print("=" * 88)
    print(f"ArtifactNet v4 (100% From Scratch) | Params: {n_params / 1e6:.2f}M | Device: {gpu_label}")
    print(
        f"Epochs: {start_epoch:02d}->{args.epochs:02d} | Micro-Batch: {batch_size} ({batch_size//2} pairs) x Accum {grad_accum} = Eff {batch_size*grad_accum} | "
        f"Steps/Epoch: {len(train_loader)} | Precision: {'BFloat16+TF32' if use_amp else 'Float32'}"
    )
    print("=" * 88)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_sum = 0.0
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            torch.cuda.reset_peak_memory_stats(device)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]")
        for step, (x_global, x_rich, x_poor, targets) in enumerate(pbar, start=1):
            x_global = x_global.to(device, non_blocking=True)
            x_rich = x_rich.to(device, non_blocking=True)
            x_poor = x_poor.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_amp):
                final_logit, details = model(
                    x_rich,
                    x_poor,
                    x_global=x_global,
                    targets=targets,
                    paired_batch=True,
                    return_details=True,
                )
                raw_loss = criterion(final_logit, details, targets, paired_batch=True)
                loss = raw_loss / grad_accum

            loss.backward()
            train_loss_sum += float(raw_loss.item())

            if step % grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(model)

            if step % 20 == 0 or step == len(train_loader):
                vram_gb = (torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if use_amp else 0.0
                pbar.set_postfix(
                    loss=f"{train_loss_sum / step:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    VRAM=f"{vram_gb:.1f}GB",
                )

        avg_train_loss = train_loss_sum / max(1, len(train_loader))

        # --- Validation 1: Full Clean Hold-Out Val Split ---
        ema.ema.eval()
        val_loss_sum = 0.0
        all_t, all_p = [], []
        stream_probs = [[] for _ in range(len(STREAM_KEYS))]
        d_real_r, d_real_a = [], []

        with torch.no_grad():
            for x_global, x_rich, x_poor, targets in tqdm(
                val_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Val]  "
            ):
                x_global = x_global.to(device, non_blocking=True)
                x_rich = x_rich.to(device, non_blocking=True)
                x_poor = x_poor.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_amp):
                    final_logit, details = ema.ema(
                        x_rich, x_poor, x_global=x_global, return_details=True
                    )
                    v_loss = criterion(final_logit, details, targets, paired_batch=False)

                val_loss_sum += float(v_loss.item())
                t_np = targets.cpu().numpy()
                all_t.extend(t_np)
                all_p.extend(torch.sigmoid(final_logit.to(torch.float32)).squeeze(1).cpu().numpy())

                d_np = details["d_real"].to(torch.float32).squeeze(1).cpu().numpy()
                d_real_r.extend(d_np[t_np < 0.5])
                d_real_a.extend(d_np[t_np >= 0.5])

                for idx, sk in enumerate(STREAM_KEYS):
                    stream_probs[idx].extend(torch.sigmoid(details[sk].to(torch.float32)).squeeze(1).cpu().numpy())

        y_true = np.array(all_t)
        clean_metrics = compute_metrics(y_true, np.array(all_p))
        stream_f1s = [compute_metrics(y_true, np.array(sp))["f1"] * 100.0 for sp in stream_probs]

        # --- Validation 2: Cross-Domain Scale / Codec / ISP Perturbation Suite ---
        pert_suite = evaluate_perturbation_suite(
            ema.ema,
            val_df,
            data_dir,
            device,
            samples_per_mode=240,
            batch_size=batch_size,
            num_workers=max(2, args.workers // 2),
        )
        pert_agg = pert_suite["aggregate_perturbed"]

        clean_f1 = clean_metrics["f1"]
        pert_f1 = pert_agg["f1"]
        gen_score = (2.0 * clean_f1 * pert_f1) / (clean_f1 + pert_f1 + 1e-8)
        elapsed = time.time() - t0

        mean_dr_real = float(np.mean(d_real_r)) if d_real_r else 0.0
        mean_dr_ai = float(np.mean(d_real_a)) if d_real_a else 0.0

        print(
            f"[Epoch {epoch:02d}/{args.epochs:02d}] ({elapsed:.0f}s) | "
            f"Loss: {avg_train_loss:.4f}/{val_loss_sum / max(1, len(val_loader)):.4f} | "
            f"Clean F1: {clean_f1*100:.2f}% (AUC {clean_metrics['roc_auc']*100:.2f}%) | "
            f"Perturbed OOD F1: {pert_f1*100:.2f}% (Spec {pert_agg['specificity']*100:.1f}% / Rec {pert_agg['recall']*100:.1f}%) | "
            f"GenScore: {gen_score*100:.2f}%"
        )
        print(
            f"   Streams F1 -> S1_Micro: {stream_f1s[0]:.1f}% | S2_WaveletFFT: {stream_f1s[1]:.1f}% | "
            f"S3_GlobalViT: {stream_f1s[2]:.1f}% | S4_GramPhys: {stream_f1s[3]:.1f}% | "
            f"Manifold d_real (Real vs AI): {mean_dr_real:.3f} vs {mean_dr_ai:.3f}"
        )

        row_data = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss_sum / max(1, len(val_loader)),
            "val_f1": clean_f1,
            "val_accuracy": clean_metrics["accuracy"],
            "val_specificity": clean_metrics["specificity"],
            "val_recall": clean_metrics["recall"],
            "val_roc_auc": clean_metrics["roc_auc"],
            "perturbed_f1": pert_f1,
            "perturbed_specificity": pert_agg["specificity"],
            "perturbed_recall": pert_agg["recall"],
            "perturbed_roc_auc": pert_agg["roc_auc"],
            "gen_score": gen_score,
            "d_real_photo": mean_dr_real,
            "d_real_ai": mean_dr_ai,
            "s1_f1": stream_f1s[0],
            "s2_f1": stream_f1s[1],
            "s3_f1": stream_f1s[2],
            "s4_f1": stream_f1s[3],
            "time_sec": elapsed,
        }
        history.append(row_data)
        history_df = pd.DataFrame(history)
        history_df.to_csv(history_csv, index=False)
        save_training_curves(history_df, assets_dir / "training_curves_v4.png")

        best_clean_f1 = max(best_clean_f1, clean_f1)
        is_new_best = gen_score > best_gen_score
        if is_new_best:
            best_gen_score = gen_score

        last_payload = {
            "epoch": epoch,
            "architecture": "ArtifactNetV4",
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "clean_metrics": clean_metrics,
            "perturbed_metrics": pert_agg,
            "best_gen_score": best_gen_score,
            "best_clean_f1": best_clean_f1,
        }
        torch.save(last_payload, last_v4_path)

        if is_new_best:
            best_payload = {
                "epoch": epoch,
                "architecture": "ArtifactNetV4",
                "model_state_dict": ema.ema.state_dict(),
                "metrics": clean_metrics,
                "perturbed_metrics": pert_agg,
                "val_f1": clean_f1,
                "perturbed_f1": pert_f1,
                "gen_score": gen_score,
            }
            torch.save(best_payload, best_v4_path)
            shutil.copy2(best_v4_path, best_generic_path)
            print(
                f"   >>> New Best Generalization Checkpoint Saved! "
                f"(GenScore: {gen_score*100:.2f}% | Clean F1: {clean_f1*100:.2f}% | Perturbed F1: {pert_f1*100:.2f}%)\n"
            )

    # --- Post-Training Multi-Condition Temperature, Manifold & Statistical Physics Calibration ---
    if best_v4_path.exists():
        print("\n" + "=" * 88)
        print("[Post-Train Calibration] Fitting Multi-Condition Temperature, Manifold Blend & Physics Weights...")
        best_ckpt = torch.load(best_v4_path, map_location=device, weights_only=False)
        cal_model = ArtifactNetV4(dropout=0.0).to(device)
        cal_model.load_state_dict(best_ckpt["model_state_dict"], strict=True)

        cal_summary = fit_post_training_calibration(
            cal_model,
            val_df,
            data_dir,
            device,
            samples_per_cond=600,
            batch_size=batch_size,
            num_workers=max(2, args.workers // 2),
        )
        best_ckpt["model_state_dict"] = cal_model.state_dict()
        best_ckpt["calibration"] = cal_summary
        torch.save(best_ckpt, best_v4_path)
        shutil.copy2(best_v4_path, best_generic_path)

        print(
            f"[Post-Train Calibration] Complete! "
            f"blend_alpha={cal_summary['blend_alpha']:.2f}, temp={cal_summary['calib_temp']:.3f}, "
            f"bias={cal_summary['calib_bias']:+.3f} | "
            f"5-Condition F1={cal_summary['multi_cond_f1']*100:.2f}% "
            f"(Spec={cal_summary['multi_cond_specificity']*100:.1f}%, Rec={cal_summary['multi_cond_recall']*100:.1f}%)"
        )
        print("=" * 88)
        print("Next Steps:")
        print("  1. Full Hold-Out & 7-Part OOD Stress Evaluation : python src/evaluate.py")
        print("  2. Launch Interactive Gradio XAI Inspector      : python app.py")
        print("=" * 88)


if __name__ == "__main__":
    train()
