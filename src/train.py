import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# უზრუნველყოფს სუფთა იმპორტს როგორც PyCharm-იდან პირდაპირ გაშვებისას, ისე ტერმინალიდან
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import create_dataloaders
from model import ArtifactNet
from utils import compute_metrics, seed_everything


# =====================================================================
# კონფიგურაცია (RTX 5060 Ti 16GB + i5-12600KF)
# =====================================================================
CONFIG = {
    "seed": 42,
    "img_size": 256,
    "batch_size": 80,       # ავსებს ~13-14 GB VRAM-ს
    "num_workers": 6,        # i5-12600KF-ის 8 ნაკადი სწრაფი წაკითხვისთვის
    "epochs": 15,
    "max_lr": 1e-3,
    "weight_decay": 1e-2,
    "grad_clip": 1.0,
    "label_smoothing": 0.05,
}


def smooth_labels(targets: torch.Tensor, smoothing: float = 0.05) -> torch.Tensor:
    """ამცირებს მოდელის ზედმეტ თავდაჯერებულობას (Overfitting-ის პრევენცია)."""
    if smoothing <= 0.0:
        return targets
    return targets * (1.0 - smoothing) + 0.5 * smoothing


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    """ატარებს სწავლების 1 სრულ ეპოქას BFloat16 აჩქარებით."""
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{total_epochs:02d} [TRAIN]", leave=True)

    for step, (images, labels) in enumerate(pbar, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)
        smoothed_targets = smooth_labels(labels, CONFIG["label_smoothing"])

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images)
            loss = criterion(logits, smoothed_targets)

        loss.backward()

        # გრადიენტების შეზღუდვა სტაბილური სწავლებისთვის
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["grad_clip"])

        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        avg_loss = running_loss / step

        # ლაივ მონაცემები პროგრეს-ბარში (Loss, LR და VRAM მოხმარება)
        if step % 10 == 0 or step == len(loader):
            vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lr": f"{current_lr:.2e}",
                "VRAM": f"{vram_gb:.1f}GB"
            })

    return running_loss / len(loader)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, dict]:
    """ამოწმებს მოდელს ვალიდაციის სეტზე და ითვლის F1, Accuracy, ROC-AUC მეტრიკებს."""
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_labels = []

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{total_epochs:02d} [VAL]  ", leave=True)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels_gpu = labels.to(device, non_blocking=True).unsqueeze(1)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images)
            loss = criterion(logits, labels_gpu)
            probs = torch.sigmoid(logits)

        running_loss += loss.item()
        all_probs.append(probs.float().cpu().numpy())
        all_labels.append(labels.numpy())

    val_loss = running_loss / len(loader)
    y_probs = np.concatenate(all_probs).flatten()
    y_true = np.concatenate(all_labels).flatten()

    metrics = compute_metrics(y_true, y_probs, threshold=0.5)
    return val_loss, metrics


def main() -> None:
    seed_everything(CONFIG["seed"])

    # Blackwell / RTX 50-series აპარატურული ოპტიმიზაციების ჩართვა
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = SRC_DIR.parent
    data_dir = project_root / "data"
    checkpoints_dir = project_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"🔥 ვიწყებთ ArtifactNet-ის სწავლებას | მოწყობილობა: {torch.cuda.get_device_name(0)}")
    print(f"📦 Batch Size: {CONFIG['batch_size']} | რეზოლუცია: {CONFIG['img_size']}x{CONFIG['img_size']} | ეპოქები: {CONFIG['epochs']}")
    print("=" * 65)

    # 1. DataLoader-ების მომზადება
    train_loader, val_loader, _ = create_dataloaders(
        data_dir=data_dir,
        img_size=CONFIG["img_size"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
    )
    print(f"✅ მონაცემები მზადაა: Train={len(train_loader.dataset):,} ფოტო | Val={len(val_loader.dataset):,} ფოტო\n")

    # 2. მოდელის, Loss-ის, Optimizer-ისა და Scheduler-ის ინიციალიზაცია
    model = ArtifactNet(num_classes=1, dropout_rate=0.3).to(device)
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["max_lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG["max_lr"],
        epochs=CONFIG["epochs"],
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
    )

    best_f1 = 0.0
    history = []

    # 3. მთავარი სასწავლო ციკლი
    for epoch in range(1, CONFIG["epochs"] + 1):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, CONFIG["epochs"]
        )
        val_loss, metrics = validate(
            model, val_loader, criterion, device, epoch, CONFIG["epochs"]
        )

        elapsed = time.time() - start_time

        print(
            f"📊 შედეგი [Epoch {epoch:02d}/{CONFIG['epochs']:02d}] ({elapsed:.0f}წმ) -> "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"F1: {metrics['f1']*100:.2f}% | Acc: {metrics['accuracy']*100:.2f}% | "
            f"AUC: {metrics['roc_auc']*100:.2f}%"
        )

        # ვინახავთ ისტორიას CSV-სთვის
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_f1": round(metrics["f1"], 5),
            "val_accuracy": round(metrics["accuracy"], 5),
            "val_precision": round(metrics["precision"], 5),
            "val_recall": round(metrics["recall"], 5),
            "val_roc_auc": round(metrics["roc_auc"], 5),
            "time_sec": round(elapsed, 1),
        }
        history.append(epoch_record)
        pd.DataFrame(history).to_csv(checkpoints_dir / "training_history.csv", index=False)

        # ვინახავთ ბოლო მოდელს
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_f1": best_f1,
            "config": CONFIG,
        }
        torch.save(checkpoint_data, checkpoints_dir / "last_model.pth")

        # თუ ახალი რეკორდია F1-Score-ში, ვინახავთ როგორც საუკეთესო მოდელს!
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            checkpoint_data["best_f1"] = best_f1
            torch.save(checkpoint_data, checkpoints_dir / "best_model.pth")
            print(f"🏆 ახალი საუკეთესო მოდელი შენახულია! (F1: {best_f1*100:.2f}%)\n")
        else:
            print()

    print("=" * 65)
    print(f"🎉 სწავლება დასრულდა! საუკეთესო Validation F1-Score: {best_f1*100:.2f}%")
    print(f"💾 მოდელი შენახულია ფოლდერში: {checkpoints_dir / 'best_model.pth'}")
    print("=" * 65)


if __name__ == "__main__":
    main()