import os
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def seed_everything(seed: int = 42) -> None:
    """აფიქსირებს ყველა შემთხვევითობის გენერატორს (Reproducibility)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def compute_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> dict:
    """ითვლის კონკურსის მთავარ მეტრიკას (F1-Score) და დამატებით მაჩვენებლებს."""
    y_pred = (y_probs >= threshold).astype(int)

    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.0,
    }