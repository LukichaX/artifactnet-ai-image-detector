import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, roc_curve
from tqdm import tqdm

from src.dataset import create_dataloaders, smash_and_reconstruct_rich_poor
from src.model import ArtifactNetV2
from src.utils import compute_metrics, seed_everything


@torch.no_grad()
def predict_single_image_tta(model: torch.nn.Module, img_rgb: np.ndarray, device: torch.device) -> float:
    """
    Test-Time Augmentation (TTA) 1:1 პიქსელებზე:
    ამოწმებს ორიგინალსა და ჰორიზონტალურად შებრუნებულ Rich/Poor წყვილს.
    """
    xr1, xp1 = smash_and_reconstruct_rich_poor(img_rgb, out_size=256, patch_size=32, is_train=False)
    img_flip = cv2.flip(img_rgb, 1)
    xr2, xp2 = smash_and_reconstruct_rich_poor(img_flip, out_size=256, patch_size=32, is_train=False)

    xr = torch.from_numpy(np.stack([xr1, xr2]).transpose(0, 3, 1, 2)).to(device, dtype=torch.float32) / 255.0
    xp = torch.from_numpy(np.stack([xp1, xp2]).transpose(0, 3, 1, 2)).to(device, dtype=torch.float32) / 255.0

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(xr, xp)
        probs = torch.sigmoid(logits).squeeze(1)
    return float(probs.mean().item())


def evaluate() -> None:
    seed_everything(42)
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    ckpt_path = project_root / "checkpoints" / "best_model_v2.pth"
    assets_dir = project_root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ArtifactNetV2(dropout=0.0).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"🔍 ვამოწმებთ ArtifactNet v2.5 Ultra-ს (Epoch {ckpt.get('epoch', '?')}) ხელუხლებელ Test სეტზე...")
    _, _, test_loader = create_dataloaders(data_dir=data_dir, img_size=256, batch_size=64, num_workers=8)

    all_targets, all_probs = [], []
    with torch.no_grad():
        for x_rich, x_poor, targets in tqdm(test_loader, desc="Testing"):
            x_rich = x_rich.to(device, non_blocking=True)
            x_poor = x_poor.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x_rich, x_poor)
                probs = torch.sigmoid(logits).squeeze(1)
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.to(torch.float32).cpu().numpy())

    y_true = np.array(all_targets)
    y_probs = np.array(all_probs)
    metrics = compute_metrics(y_true, y_probs)

    print("\n🏆 საბოლოო შედეგი ხელუხლებელ Test Set-ზე:")
    print(f"   • F1-Score:  {metrics['f1']*100:.2f}%")
    print(f"   • Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"   • Precision: {metrics['precision']*100:.2f}%")
    print(f"   • Recall:    {metrics['recall']*100:.2f}%")
    print(f"   • ROC-AUC:   {metrics['roc_auc']*100:.2f}%")

    # Confusion Matrix + ROC Curve
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
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
    axes[0].set_title("Confusion Matrix (ArtifactNet v2.5 Ultra)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    axes[1].plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']*100:.2f}%", linewidth=2.2)
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(assets_dir / "test_evaluation_v2.png", dpi=200)
    plt.close()

    # Kaggle test_data_v2 OOD პროგნოზირება
    test_csv = data_dir / "test.csv"
    if test_csv.exists():
        sub_df = pd.read_csv(test_csv)
        preds = []
        for _, row in tqdm(sub_df.iterrows(), total=len(sub_df), desc="Predicting test_data_v2 (TTA)"):
            img_path = data_dir / row["id"]
            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                preds.append(0)
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            prob = predict_single_image_tta(model, img_rgb, device)
            preds.append(1 if prob >= 0.5 else 0)

        sub_df["label"] = preds
        sub_df.to_csv(project_root / "submission_v2.csv", index=False)
        n_ai = int(np.sum(preds))
        n_real = len(preds) - n_ai
        print(f"✅ Kaggle submission_v2.csv მზადაა! (Real: {n_real}, AI: {n_ai} [{n_ai/len(preds)*100:.1f}%])")


if __name__ == "__main__":
    evaluate()