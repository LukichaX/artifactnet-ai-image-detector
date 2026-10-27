import os
import sys
import time
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.dataset import create_dataloaders, smash_and_reconstruct_rich_poor
from src.model import ArtifactNetV2
from src.utils import AsymmetricHypersphereLoss, ModelEMA, compute_metrics, seed_everything


@torch.no_grad()
def run_ood_probe(model: torch.nn.Module, data_dir: Path, device: torch.device, sample_size: int = 400) -> float:
    test_csv = data_dir / "test.csv"
    if not test_csv.exists():
        return 0.0

    df = pd.read_csv(test_csv).head(sample_size)
    model.eval()
    probs = []

    for _, row in df.iterrows():
        img_path = data_dir / row["id"]
        if not img_path.exists():
            img_path = data_dir / "test_data_v2" / Path(row["id"]).name
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        xr_np, xp_np = smash_and_reconstruct_rich_poor(img_rgb, out_size=256, patch_size=32, is_train=False)
        xr = torch.from_numpy(np.ascontiguousarray(xr_np.transpose(2, 0, 1))).unsqueeze(0).to(device)
        xp = torch.from_numpy(np.ascontiguousarray(xp_np.transpose(2, 0, 1))).unsqueeze(0).to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logit = model(xr, xp)
            prob = torch.sigmoid(logit).item()
        probs.append(prob)

    if not probs:
        return 0.0
    probs_arr = np.array(probs)
    return float((probs_arr >= 0.5).mean() * 100.0)


def train() -> None:
    seed_everything(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data_dir = PROJECT_ROOT / "data"
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32          # 32 წყვილი = 64 სურათი თითო ნაბიჯზე (~7.5 GB VRAM, 0 GB Shared RAM)
    grad_accum_steps = 2     # ეფექტური Batch Size = 32 * 2 = 64 წყვილი (128 სურათი)
    img_size = 256
    epochs = 20              # 20 სრული ეპოქა (~2.5 საათი)
    max_lr = 8e-4

    print(f"🔥 ვიწყებთ ArtifactNet v2.5 Ultra-ს სწავლებას | მოწყობილობა: {torch.cuda.get_device_name(0)}")
    train_loader, val_loader, _ = create_dataloaders(
        data_dir=data_dir, img_size=img_size, batch_size=batch_size, num_workers=4
    )

    model = ArtifactNetV2(dropout=0.25).to(device)
    ema = ModelEMA(model, decay=0.998)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 მოდელის პარამეტრები: {total_params:,} (~{total_params / 1e6:.2f}M — 100% From Scratch)")
    print(f"📦 Micro-Batch: {batch_size} (x2 Rich/Poor = 64 ფოტო) | Grad Accum: {grad_accum_steps} | ეპოქები: {epochs}")

    criterion = AsymmetricHypersphereLoss(smoothing=0.05, sphere_weight=0.35, margin_mult=2.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=1e-2)
    steps_per_epoch = len(train_loader) // grad_accum_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    best_f1 = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_sum = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [TRAIN]")
        for step, (x_rich, x_poor, targets) in enumerate(pbar, start=1):
            x_rich = x_rich.to(device, non_blocking=True)
            x_poor = x_poor.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, dist_sq, radius_sq = model(x_rich, x_poor, targets=targets, return_aux=True)
                raw_loss = criterion(logits, dist_sq, radius_sq, targets)
                loss = raw_loss / grad_accum_steps

            loss.backward()
            train_loss_sum += raw_loss.item()

            if step % grad_accum_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if step // grad_accum_steps <= steps_per_epoch:
                    scheduler.step()
                ema.update(model)

            if step % 10 == 0 or step == len(train_loader):
                vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                pbar.set_postfix(
                    loss=f"{train_loss_sum / step:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    VRAM=f"{vram_gb:.1f}GB",
                )

        avg_train_loss = train_loss_sum / len(train_loader)

        ema.ema.eval()
        val_loss_sum = 0.0
        all_targets, all_probs = [], []

        with torch.no_grad():
            for x_rich, x_poor, targets in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [VAL]  "):
                x_rich = x_rich.to(device, non_blocking=True)
                x_poor = x_poor.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits, dist_sq, radius_sq = ema.ema(x_rich, x_poor, return_aux=True)
                    v_loss = criterion(logits, dist_sq, radius_sq, targets)
                    probs = torch.sigmoid(logits).squeeze(1)

                val_loss_sum += v_loss.item()
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.to(torch.float32).cpu().numpy())

        avg_val_loss = val_loss_sum / len(val_loader)
        metrics = compute_metrics(np.array(all_targets), np.array(all_probs))
        ood_ai_pct = run_ood_probe(ema.ema, data_dir, device, sample_size=400)
        elapsed = time.time() - t0

        print(
            f"📊 შედეგი [Epoch {epoch:02d}/{epochs:02d}] ({elapsed:.0f}წმ) -> "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"F1: {metrics['f1']*100:.2f}% | AUC: {metrics['roc_auc']*100:.2f}% | "
            f"OOD AI Rate: {ood_ai_pct:.1f}%"
        )

        row_data = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_f1": metrics["f1"],
            "val_accuracy": metrics["accuracy"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_roc_auc": metrics["roc_auc"],
            "ood_ai_ratio_pct": ood_ai_pct,
            "time_sec": elapsed,
        }
        history.append(row_data)
        pd.DataFrame(history).to_csv(ckpt_dir / "training_history_v2.csv", index=False)

        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": ema.ema.state_dict(),
            "metrics": metrics,
            "ood_ai_ratio_pct": ood_ai_pct,
        }
        torch.save(ckpt_payload, ckpt_dir / "last_model_v2.pth")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(ckpt_payload, ckpt_dir / "best_model_v2.pth")
            print(f"🏆 ახალი საუკეთესო v2.5 მოდელი შენახულია! (F1: {best_f1*100:.2f}% | OOD AI: {ood_ai_pct:.1f}%)")


if __name__ == "__main__":
    train()