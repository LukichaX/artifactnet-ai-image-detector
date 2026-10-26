import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import create_dataloaders, get_transforms
from model import ArtifactNet
from utils import compute_metrics


class KaggleTestDataset(Dataset):
    """კითხულობს test.csv-ის 5,540 სურათს Kaggle-ის პროგნოზებისთვის."""
    def __init__(self, df: pd.DataFrame, data_dir: Path, transform):
        self.data_dir = data_dir
        self.ids = df["id"].values
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        rel_path = self.ids[idx]
        img_path = str(self.data_dir / rel_path)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"ვერ მოიძებნა: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor_img = self.transform(image=image)["image"]
        return tensor_img, rel_path


def plot_training_history(history_csv: Path, assets_dir: Path) -> None:
    """ხატავს სწავლების გრაფიკებს GitHub README-სთვის."""
    if not history_csv.exists():
        return
    df = pd.read_csv(history_csv)
    best_ep = int(df.loc[df["val_f1"].idxmax(), "epoch"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    axes[0].plot(df["epoch"], df["train_loss"], marker="o", linewidth=2, label="Train Loss")
    axes[0].plot(df["epoch"], df["val_loss"], marker="s", linewidth=2, label="Val Loss")
    axes[0].axvline(best_ep, linestyle="--", color="green", alpha=0.7, label=f"Best Epoch ({best_ep})")
    axes[0].set_title("Training vs Validation Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(df["epoch"], df["val_f1"] * 100, marker="o", linewidth=2, label="Val F1 (%)")
    axes[1].plot(df["epoch"], df["val_accuracy"] * 100, marker="^", linestyle="-.", label="Val Acc (%)")
    axes[1].plot(df["epoch"], df["val_roc_auc"] * 100, marker="d", linewidth=2, label="Val ROC-AUC (%)")
    axes[1].set_title("Validation Metrics (%)", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score (%)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(assets_dir / "training_curves.png")
    plt.close()


def plot_evaluation_charts(y_true: np.ndarray, y_probs: np.ndarray, assets_dir: Path) -> None:
    """ხატავს Confusion Matrix-ს და ROC Curve-ს."""
    y_pred = (y_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real (0)", "AI (1)"],
        yticklabels=["Real (0)", "AI (1)"],
        ax=axes[0],
        annot_kws={"size": 13, "weight": "bold"},
    )
    axes[0].set_title("Confusion Matrix (Hold-out Test Set)", fontweight="bold")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    axes[1].plot(fpr, tpr, color="#ea580c", linewidth=2.5, label="ArtifactNet ROC")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--")
    axes[1].set_title("ROC Curve (Test Set)", fontweight="bold")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(assets_dir / "test_evaluation.png")
    plt.close()


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = SRC_DIR.parent
    data_dir = project_root / "data"
    checkpoints_dir = project_root / "checkpoints"
    assets_dir = project_root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoints_dir / "best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {"img_size": 256, "batch_size": 80, "num_workers": 4})

    model = ArtifactNet(num_classes=1, dropout_rate=0.0).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("=" * 65)
    print(f"🔍 ვამოწმებთ საუკეთესო მოდელს (Epoch {checkpoint['epoch']}) დამოუკიდებელ Test სეტზე")
    print("=" * 65)

    _, _, test_loader = create_dataloaders(
        data_dir=data_dir,
        img_size=config["img_size"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    all_probs, all_labels = [], []
    for images, labels in tqdm(test_loader, desc="Testing [7,996 images]"):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images)
            probs = torch.sigmoid(logits)
        all_probs.append(probs.float().cpu().numpy())
        all_labels.append(labels.numpy())

    y_probs = np.concatenate(all_probs).flatten()
    y_true = np.concatenate(all_labels).flatten()
    metrics = compute_metrics(y_true, y_probs, threshold=0.5)

    print("\n🏆 საბოლოო შედეგი ხელუხლებელ Test Set-ზე (7,996 ფოტო):")
    print(f"   • F1-Score:  {metrics['f1']*100:.2f}%")
    print(f"   • Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"   • Precision: {metrics['precision']*100:.2f}%")
    print(f"   • Recall:    {metrics['recall']*100:.2f}%")
    print(f"   • ROC-AUC:   {metrics['roc_auc']*100:.2f}%")

    plot_training_history(checkpoints_dir / "training_history.csv", assets_dir)
    plot_evaluation_charts(y_true, y_probs, assets_dir)
    print(f"📈 გრაფიკები შენახულია ფოლდერში: {assets_dir}")

    # Kaggle-ის 5,540 უცნობ სურათზე პროგნოზირება
    test_csv_path = data_dir / "test.csv"
    if test_csv_path.exists():
        test_df = pd.read_csv(test_csv_path)
        _, val_tf = get_transforms(img_size=config["img_size"])
        kaggle_ds = KaggleTestDataset(test_df, data_dir, transform=val_tf)
        kaggle_loader = DataLoader(kaggle_ds, batch_size=config["batch_size"], shuffle=False, num_workers=4)

        preds_list, ids_list = [], []
        for images, img_ids in tqdm(kaggle_loader, desc="Predicting test_data_v2"):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                probs = torch.sigmoid(model(images)).float().cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)
            preds_list.extend(preds.tolist())
            ids_list.extend(list(img_ids))

        sub_df = pd.DataFrame({"id": ids_list, "label": preds_list})
        sub_path = project_root / "submission.csv"
        sub_df.to_csv(sub_path, index=False)
        ai_count = int((sub_df["label"] == 1).sum())
        real_count = int((sub_df["label"] == 0).sum())
        print(f"✅ Kaggle submission.csv მზადაა! (Real: {real_count}, AI: {ai_count})")


if __name__ == "__main__":
    main()