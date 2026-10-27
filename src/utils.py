import copy
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class AsymmetricHypersphereLoss(nn.Module):
    """
    SimLBR (CVPR 2026) + DeepSVDD ჰიბრიდული შეცდომის ფუნქცია (დაზღვეული რადიუსის კოლაფსისგან):
    - რეალური ფოტოები (y=0) იკუმშება c_real ცენტრის გარშემო (dist_sq <= R^2),
    - AI და Pseudo-Fake ფოტოები (y=1) განიზიდება სფეროს გარეთ მინიმუმ target_margin მანძილზე,
      სადაც radius_sq.detach() ბლოკავს რადიუსის ხელოვნურ განულებას.
    """
    def __init__(self, smoothing: float = 0.05, sphere_weight: float = 0.35, margin_mult: float = 2.5) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.sphere_weight = sphere_weight
        self.margin_mult = margin_mult

    def forward(
        self, logits: torch.Tensor, dist_sq: torch.Tensor, radius_sq: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        targets_col = targets.view(-1, 1).to(logits.dtype)
        smoothed_targets = targets_col * (1.0 - self.smoothing) + 0.5 * self.smoothing
        bce_loss = F.binary_cross_entropy_with_logits(logits, smoothed_targets)

        is_real = (targets_col < 0.5).to(dist_sq.dtype)
        is_fake = 1.0 - is_real

        real_pull = is_real * (dist_sq + 2.0 * F.relu(dist_sq - radius_sq))
        target_margin = torch.clamp(self.margin_mult * radius_sq.detach(), min=1.0)
        fake_push = is_fake * F.relu(target_margin - dist_sq)

        sphere_loss = torch.mean(real_pull + fake_push) + 0.005 * radius_sq
        return bce_loss + self.sphere_weight * sphere_loss


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.998) -> None:
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, ema_v in self.ema.state_dict().items():
            model_v = msd[k].detach()
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)
            else:
                ema_v.copy_(model_v)


def compute_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_probs >= threshold).astype(int)
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 0.0,
    }