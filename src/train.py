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

from src.dataset import create_dataloaders_v5, prepare_splits_v5
from src.model import ArtifactNetV5
from src.utils import (
    STREAM_KEYS,
    ArtifactNetV5Loss,
    ModelEMA,
    compute_metrics,
    evaluate_perturbation_suite,
    fit_post_training_calibration,
    seed_everything,
)


def backup_legacy_checkpoints(ckpt_dir: Path) -> None:
    """Safely archives any pre-v5 checkpoints into `checkpoints/archive_v4/` before saving v5 weights."""
    best_generic = ckpt_dir / "best_model.pth"
    if not best_generic.exists() and not (ckpt_dir / "best_model_v4.pth").exists():
        return
    arch = ""
    if best_generic.exists():
        try:
            meta = torch.load(best_generic, map_location="cpu", weights_only=False)
            arch = meta.get("architecture", "") if isinstance(meta, dict) else ""
        except Exception:
            arch = ""

    if arch == "ArtifactNetV5":
        return

    archive_dir = ckpt_dir / "archive_v4"
    archive_dir.mkdir(parents=True, exist_ok=True)
    legacy_names = [
        "best_model.pth",
        "best_model_v4.pth",
        "last_model_v4.pth",
        "training_history_v4.csv",
    ]
    copied = []
    for name in legacy_names:
        src = ckpt_dir / name
        dst = archive_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied.append(name)
    if copied:
        print(f"[Backup] Archived v4 checkpoints to {archive_dir}: {', '.join(copied)}")


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
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))

    ep = history_df["epoch"]
    axes[0].plot(ep, history_df["train_loss"], marker="o", label="Train Loss")
    axes[0].plot(ep, history_df["val_id_loss"], marker="s", label="Val-ID Loss")
    axes[0].plot(ep, history_df["val_ood_loss"], marker="^", label="Unseen Val-OOD Loss")
    axes[0].set_title("Multi-Generator & OOD Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(ep, history_df["val_id_f1"] * 100.0, marker="o", label="Val-ID F1 (%)")
    axes[1].plot(ep, history_df["val_ood_f1"] * 100.0, marker="s", label="Unseen-Gen OOD F1 (%)")
    axes[1].plot(ep, history_df["ood_laundered_f1"] * 100.0, marker="d", label="Internet-Laundered OOD F1 (%)")
    axes[1].plot(ep, history_df["gen_score"] * 100.0, marker="^", linewidth=2.3, label="Harmonic GenScore (%)")
    axes[1].set_title("Multi-Generator & Internet-Laundered Generalization")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 Score (%)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(ep, history_df["ood_laundered_spec"] * 100.0, marker="o", label="Laundered OOD Real Spec (%)")
    axes[2].plot(ep, history_df["ood_laundered_rec"] * 100.0, marker="s", label="Laundered OOD AI Recall (%)")
    axes[2].plot(ep, history_df["val_ood_auc"] * 100.0, marker="^", label="Unseen-Gen OOD ROC-AUC (%)")
    axes[2].set_title("Unseen-Generator Specificity & Recall")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Percentage (%)")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


@torch.no_grad()
def _run_eval_loader(model, loader, criterion, device, dev_type: str, use_amp: bool) -> dict:
    model.eval()
    loss_sum = 0.0
    all_t, all_p = [], []
    fam_correct, fam_total = 0, 0
    stream_probs = [[] for _ in range(len(STREAM_KEYS))]
    d_real_r, d_real_a = [], []

    for batch in loader:
        if len(batch) == 5:
            x_global, x_rich, x_poor, targets, fam_targets = batch
            fam_targets = fam_targets.to(device, non_blocking=True)
        else:
            x_global, x_rich, x_poor, targets = batch
            fam_targets = None

        x_global = x_global.to(device, non_blocking=True)
        x_rich = x_rich.to(device, non_blocking=True)
        x_poor = x_poor.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_amp):
            final_logit, details = model(x_rich, x_poor, x_global=x_global, return_details=True)
            v_loss = criterion(
                final_logit, details, targets, family_targets=fam_targets, paired_batch=False
            )

        loss_sum += float(v_loss.item())
        t_np = targets.cpu().numpy()
        all_t.extend(t_np)
        all_p.extend(torch.sigmoid(final_logit.to(torch.float32)).squeeze(1).cpu().numpy())

        if fam_targets is not None and "family_logits" in details:
            preds_f = details["family_logits"].argmax(dim=1)
            fam_correct += int((preds_f == fam_targets).sum().item())
            fam_total += int(fam_targets.numel())

        d_np = details["d_real"].to(torch.float32).squeeze(1).cpu().numpy()
        d_real_r.extend(d_np[t_np < 0.5])
        d_real_a.extend(d_np[t_np >= 0.5])

        for idx, sk in enumerate(STREAM_KEYS):
            stream_probs[idx].extend(torch.sigmoid(details[sk].to(torch.float32)).squeeze(1).cpu().numpy())

    y_true = np.array(all_t)
    metrics = compute_metrics(y_true, np.array(all_p))
    stream_f1s = [compute_metrics(y_true, np.array(sp))["f1"] * 100.0 for sp in stream_probs]
    return {
        "loss": loss_sum / max(1, len(loader)),
        "metrics": metrics,
        "stream_f1s": stream_f1s,
        "fam_acc": (fam_correct / max(1, fam_total)) if fam_total > 0 else 0.0,
        "d_real_photo": float(np.mean(d_real_r)) if d_real_r else 0.0,
        "d_real_ai": float(np.mean(d_real_a)) if d_real_a else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ArtifactNet v5 (Multi-Generator & Internet-Resilient) From Scratch")
    parser.add_argument("--epochs", type=int, default=14, help="Total training epochs (default: 14)")
    parser.add_argument("--samples-per-epoch", type=int, default=84000, help="Stratified paired images per epoch (default: 84,000)")
    parser.add_argument("--batch-size", type=int, default=32, help="Micro-batch images per GPU pass (even; default: 32)")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps (default: 2 -> effective batch 64)")
    parser.add_argument("--lr", type=float, default=7.5e-4, help="Peak learning rate for AdamW")
    parser.add_argument("--weight-decay", type=float, default=2e-2, help="Weight decay on Conv/Linear matrices")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader CPU worker processes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints/last_model_v5.pth if available")
    parser.add_argument("--calibrate-only", action="store_true", help="Run unseen-OOD + internet-chain calibration on best_model_v5.pth")
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
    is_tty = sys.stdout.isatty()

    train_df, val_id_df, val_ood_df, _ = prepare_splits_v5(data_dir)
    best_v5_path = ckpt_dir / "best_model_v5.pth"
    best_generic_path = ckpt_dir / "best_model.pth"
    last_v5_path = ckpt_dir / "last_model_v5.pth"
    history_csv = ckpt_dir / "training_history_v5.csv"

    if args.calibrate_only:
        target_ckpt = best_v5_path if best_v5_path.exists() else best_generic_path
        if not target_ckpt.exists():
            raise FileNotFoundError(f"No trained ArtifactNetV5 checkpoint found at {target_ckpt}")
        ckpt = torch.load(target_ckpt, map_location=device, weights_only=False)
        model = ArtifactNetV5(dropout=0.0).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"[Calibrate] Fitting unseen-OOD + multi-hop internet calibration on {target_ckpt.name}...")
        cal_info = fit_post_training_calibration(
            model,
            val_id_df,
            data_dir,
            device,
            val_ood_df=val_ood_df,
            samples_per_cond=700,
            batch_size=max(args.batch_size, 48),
            num_workers=max(2, args.workers // 2),
        )
        ckpt["model_state_dict"] = model.state_dict()
        ckpt["calibration"] = cal_info
        torch.save(ckpt, best_v5_path)
        shutil.copy2(best_v5_path, best_generic_path)
        print(f"[Calibrate] Saved calibrated weights -> {best_v5_path.name} & {best_generic_path.name}")
        print(f"            {cal_info}")
        return

    batch_size = max(4, (args.batch_size // 2) * 2)
    grad_accum = max(1, args.grad_accum)

    train_ds, train_loader, val_id_loader, val_ood_loader, _ = create_dataloaders_v5(
        data_dir=data_dir,
        img_size=256,
        batch_size=batch_size,
        samples_per_epoch=args.samples_per_epoch,
        num_workers=args.workers,
        seed=args.seed,
    )

    model = ArtifactNetV5(dropout=0.25, expert_drop_prob=0.15, spectral_drop_prob=0.50).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    steps_per_epoch = max(1, len(train_loader) // grad_accum)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = min(total_steps // 6, int(1.5 * steps_per_epoch))

    optimizer, scheduler = build_optimizer_and_scheduler(
        model, lr=args.lr, weight_decay=args.weight_decay, total_steps=total_steps, warmup_steps=warmup_steps
    )
    ema = ModelEMA(model, decay=0.997)
    criterion = ArtifactNetV5Loss(smoothing=0.03)

    start_epoch = 1
    best_gen_score = 0.0
    history: list[dict] = []

    if args.resume and last_v5_path.exists():
        ckpt = torch.load(last_v5_path, map_location=device, weights_only=False)
        if ckpt.get("architecture") == "ArtifactNetV5":
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
            if history_csv.exists():
                history = pd.read_csv(history_csv).to_dict(orient="records")
            print(
                f"[Resume] Continuing ArtifactNet v5 from Epoch {start_epoch}/{args.epochs} "
                f"(Best GenScore: {best_gen_score*100:.2f}%)"
            )

    print("=" * 94)
    print(f"ArtifactNet v5 (100% From Scratch — Zero Pretrained Weights) | Params: {n_params / 1e6:.2f}M | GPU: {gpu_label}")
    print(
        f"Train Pool: {len(train_df):,} imgs | Val-ID: {len(val_id_df):,} | Unseen Val-OOD: {len(val_ood_df):,}"
    )
    print(
        f"Epochs: {start_epoch:02d}->{args.epochs:02d} | Samples/Epoch: {len(train_ds)*2:,} | "
        f"Micro-Batch: {batch_size} x Accum {grad_accum} = Eff {batch_size*grad_accum} | Steps/Epoch: {len(train_loader)}"
    )
    print("=" * 94)

    laundered_eval_modes = (
        "whatsapp_1hop",
        "social_2hop_chain",
        "webp_cdn_q75",
        "screenshot_reencode",
    )

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_ds.set_epoch(epoch)
        model.train()
        train_loss_sum = 0.0
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            torch.cuda.reset_peak_memory_stats(device)

        total_loader_steps = len(train_loader)
        iterator = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]", disable=not is_tty)

        for step, batch in enumerate(iterator, start=1):
            if len(batch) == 5:
                x_global, x_rich, x_poor, targets, fam_targets = batch
                fam_targets = fam_targets.to(device, non_blocking=True)
            else:
                x_global, x_rich, x_poor, targets = batch
                fam_targets = None

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
                raw_loss = criterion(
                    final_logit, details, targets, family_targets=fam_targets, paired_batch=True
                )
                loss = raw_loss / grad_accum

            loss.backward()
            train_loss_sum += float(raw_loss.item())

            if step % grad_accum == 0 or step == total_loader_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(model)

            if is_tty and (step % 25 == 0 or step == total_loader_steps):
                vram_gb = (torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if use_amp else 0.0
                iterator.set_postfix(
                    loss=f"{train_loss_sum / step:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    VRAM=f"{vram_gb:.1f}GB",
                )
            elif (not is_tty) and (step % 450 == 0 or step == total_loader_steps):
                vram_gb = (torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if use_amp else 0.0
                el = time.time() - t0
                ips = (step * batch_size) / max(0.1, el)
                print(
                    f"  [Ep {epoch:02d} | Step {step:04d}/{total_loader_steps:04d}] "
                    f"loss={train_loss_sum/step:.4f} | lr={optimizer.param_groups[0]['lr']:.2e} | "
                    f"speed={ips:.1f} img/s | VRAM={vram_gb:.1f}GB"
                )

        avg_train_loss = train_loss_sum / max(1, total_loader_steps)

        # --- Validation 1: In-Distribution Multi-Generator Val Split ---
        res_id = _run_eval_loader(ema.ema, val_id_loader, criterion, device, dev_type, use_amp)
        id_m = res_id["metrics"]

        # --- Validation 2: Strictly Unseen Generators & Reddit In-The-Wild (Val-OOD Clean) ---
        res_ood = _run_eval_loader(ema.ema, val_ood_loader, criterion, device, dev_type, use_amp)
        ood_m = res_ood["metrics"]

        # --- Validation 3: Internet-Laundered (1-Hop & 2-Hop) Unseen Generators ---
        pert_suite = evaluate_perturbation_suite(
            ema.ema,
            val_ood_df,
            data_dir,
            device,
            modes=laundered_eval_modes,
            samples_per_mode=260,
            batch_size=max(batch_size, 48),
            num_workers=max(2, args.workers // 2),
        )
        ood_pert_m = pert_suite["aggregate_perturbed"]

        id_f1 = max(1e-4, id_m["f1"])
        ood_f1 = max(1e-4, ood_m["f1"])
        laund_f1 = max(1e-4, ood_pert_m["f1"])
        gen_score = 1.0 / (0.25 / id_f1 + 0.375 / ood_f1 + 0.375 / laund_f1)
        elapsed = time.time() - t0

        sf1 = res_ood["stream_f1s"]
        print(
            f"[Epoch {epoch:02d}/{args.epochs:02d}] ({elapsed/60.0:.1f}m) | "
            f"Loss Tr/ID/OOD: {avg_train_loss:.4f}/{res_id['loss']:.4f}/{res_ood['loss']:.4f} | "
            f"Val-ID F1: {id_m['f1']*100:.2f}% (FamAcc {res_id['fam_acc']*100:.1f}%) | "
            f"Unseen-OOD F1: {ood_m['f1']*100:.2f}% (AUC {ood_m['roc_auc']*100:.2f}%) | "
            f"Laundered-OOD F1: {ood_pert_m['f1']*100:.2f}% (Spec {ood_pert_m['specificity']*100:.1f}%/Rec {ood_pert_m['recall']*100:.1f}%) | "
            f"GenScore: {gen_score*100:.2f}%"
        )
        print(
            f"   OOD Streams F1 -> S1_Micro: {sf1[0]:.1f}% | S2_WaveletFFT: {sf1[1]:.1f}% | "
            f"S3_MesoViT: {sf1[2]:.1f}% | S4_GramPhys: {sf1[3]:.1f}% | "
            f"OOD Manifold d_real (Real vs AI): {res_ood['d_real_photo']:.3f} vs {res_ood['d_real_ai']:.3f}"
        )

        row_data = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_id_loss": res_id["loss"],
            "val_ood_loss": res_ood["loss"],
            "val_id_f1": id_m["f1"],
            "val_id_auc": id_m["roc_auc"],
            "val_id_fam_acc": res_id["fam_acc"],
            "val_ood_f1": ood_m["f1"],
            "val_ood_acc": ood_m["accuracy"],
            "val_ood_spec": ood_m["specificity"],
            "val_ood_rec": ood_m["recall"],
            "val_ood_auc": ood_m["roc_auc"],
            "ood_laundered_f1": ood_pert_m["f1"],
            "ood_laundered_spec": ood_pert_m["specificity"],
            "ood_laundered_rec": ood_pert_m["recall"],
            "ood_laundered_auc": ood_pert_m["roc_auc"],
            "gen_score": gen_score,
            "d_real_photo": res_ood["d_real_photo"],
            "d_real_ai": res_ood["d_real_ai"],
            "s1_ood_f1": sf1[0],
            "s2_ood_f1": sf1[1],
            "s3_ood_f1": sf1[2],
            "s4_ood_f1": sf1[3],
            "time_sec": elapsed,
        }
        history.append(row_data)
        history_df = pd.DataFrame(history)
        history_df.to_csv(history_csv, index=False)
        save_training_curves(history_df, assets_dir / "training_curves_v5.png")

        is_new_best = gen_score > best_gen_score
        if is_new_best:
            best_gen_score = gen_score

        last_payload = {
            "epoch": epoch,
            "architecture": "ArtifactNetV5",
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_id_metrics": id_m,
            "val_ood_metrics": ood_m,
            "ood_laundered_metrics": ood_pert_m,
            "best_gen_score": best_gen_score,
        }
        torch.save(last_payload, last_v5_path)

        if is_new_best:
            best_payload = {
                "epoch": epoch,
                "architecture": "ArtifactNetV5",
                "model_state_dict": ema.ema.state_dict(),
                "val_id_metrics": id_m,
                "val_ood_metrics": ood_m,
                "ood_laundered_metrics": ood_pert_m,
                "gen_score": gen_score,
            }
            torch.save(best_payload, best_v5_path)
            shutil.copy2(best_v5_path, best_generic_path)
            print(
                f"   >>> New Best v5 Checkpoint Saved! "
                f"(GenScore: {gen_score*100:.2f}% | Unseen-OOD: {ood_m['f1']*100:.2f}% | Laundered-OOD: {ood_pert_m['f1']*100:.2f}%)\n"
            )

    # --- Post-Training Unseen-Generator & Multi-Hop Internet Calibration ---
    if last_v5_path.exists():
        # Compare calibrated best_v5 vs final converged EMA so uncalibrated early-epoch bias shifts never win
        last_ckpt = torch.load(last_v5_path, map_location="cpu", weights_only=False)
        best_ckpt_tmp = torch.load(best_v5_path, map_location="cpu", weights_only=False) if best_v5_path.exists() else last_ckpt
        last_auc = float(last_ckpt.get("val_ood_metrics", {}).get("roc_auc", 0.0))
        best_auc = float(best_ckpt_tmp.get("val_ood_metrics", {}).get("roc_auc", 0.0))
        if last_auc >= best_auc - 0.005:
            promoted = {
                "epoch": last_ckpt["epoch"],
                "architecture": "ArtifactNetV5",
                "model_state_dict": last_ckpt["ema_state_dict"],
                "val_id_metrics": last_ckpt.get("val_id_metrics", {}),
                "val_ood_metrics": last_ckpt.get("val_ood_metrics", {}),
                "ood_laundered_metrics": last_ckpt.get("ood_laundered_metrics", {}),
            }
            torch.save(promoted, best_v5_path)

    if best_v5_path.exists():
        print("\n" + "=" * 94)
        print("[Post-Train Calibration] Fitting Unseen-Generator + Multi-Hop Internet Temperature & Manifold Blend...")
        best_ckpt = torch.load(best_v5_path, map_location=device, weights_only=False)
        cal_model = ArtifactNetV5(dropout=0.0).to(device)
        cal_model.load_state_dict(best_ckpt["model_state_dict"], strict=True)

        cal_summary = fit_post_training_calibration(
            cal_model,
            val_id_df,
            data_dir,
            device,
            val_ood_df=val_ood_df,
            samples_per_cond=700,
            batch_size=max(batch_size, 48),
            num_workers=max(2, args.workers // 2),
        )
        best_ckpt["model_state_dict"] = cal_model.state_dict()
        best_ckpt["calibration"] = cal_summary
        torch.save(best_ckpt, best_v5_path)
        shutil.copy2(best_v5_path, best_generic_path)

        print(
            f"[Post-Train Calibration] Complete! "
            f"blend_alpha={cal_summary['blend_alpha']:.2f}, temp={cal_summary['calib_temp']:.3f}, "
            f"bias={cal_summary['calib_bias']:+.3f} | "
            f"6-Condition F1={cal_summary['multi_cond_f1']*100:.2f}% "
            f"(Spec={cal_summary['multi_cond_specificity']*100:.1f}%, Rec={cal_summary['multi_cond_recall']*100:.1f}%)"
        )
        print("=" * 94)


if __name__ == "__main__":
    train()
