import copy
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from src.dataset import SingleForensicDataset


STREAM_KEYS = [
    "logit1_micro",
    "logit2_wavelet",
    "logit3_global",
    "logit4_gram_phys",
]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


class ArtifactNetV5Loss(nn.Module):
    """
    Unified 5-Objective Forensic Generalization Loss (`ArtifactNet v5`):
      1. Smoothed Discriminative & 4-Stream Auxiliary BCE.
      2. 12-Class Generator-Family Cross-Entropy + LogSumExp Union-of-Families Margin.
      3. 16-Prototype Real-Camera Manifold Pull/Push (`d_min` separation prevents OOD collapse to Real).
      4. Pair-Synchronized Internet-Laundered Twin Margin Ranking (`(Real_k, AI_k)` under identical 1/2-hop chains).
      5. Stream Gate Load-Balancing Regularization.
    """

    def __init__(
        self,
        smoothing: float = 0.03,
        pair_margin_logit: float = 2.5,
        pair_margin_dist: float = 0.34,
        manifold_push_margin: float = 0.38,
    ) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.pair_margin_logit = pair_margin_logit
        self.pair_margin_dist = pair_margin_dist
        self.manifold_push_margin = manifold_push_margin

    def _bce(self, logits: torch.Tensor, targets_col: torch.Tensor) -> torch.Tensor:
        t_smooth = targets_col * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits.to(torch.float32), t_smooth)

    def forward(
        self,
        final_logit: torch.Tensor,
        details: dict,
        targets: torch.Tensor,
        family_targets: torch.Tensor | None = None,
        paired_batch: bool = True,
    ) -> torch.Tensor:
        dev_type = "cuda" if final_logit.is_cuda else "cpu"
        with torch.amp.autocast(device_type=dev_type, enabled=False):
            t_col = targets.view(-1, 1).to(torch.float32)
            is_real = (t_col < 0.5).to(torch.float32)
            is_fake = 1.0 - is_real

            disc_logit = details["disc_logit"].to(torch.float32)
            raw_fused = details["raw_fused_logit"].to(torch.float32)
            manifold_logit = details["manifold_logit"].to(torch.float32)
            d_min = details["d_real"].to(torch.float32)
            d_train = details["d_real_train"].to(torch.float32)
            tau = details["manifold_tau"].to(torch.float32)

            # 1. Main + Per-Stream Auxiliary BCE
            loss_main = 0.50 * self._bce(raw_fused, t_col) + 0.50 * self._bce(disc_logit, t_col)
            loss_streams = sum(self._bce(details[k], t_col) for k in STREAM_KEYS) / len(STREAM_KEYS)

            # 2. 12-Family Multi-Generator Supervision + Union-of-AI Log-Odds BCE
            loss_fam_bin = self._bce(details["family_ai_logit"], t_col)
            if family_targets is not None and "family_logits" in details:
                fam_t = family_targets.view(-1).to(torch.long)
                loss_fam_ce = F.cross_entropy(
                    details["family_logits"].to(torch.float32), fam_t, label_smoothing=0.04
                )
                loss_family = 0.65 * loss_fam_ce + 0.35 * loss_fam_bin
            else:
                loss_family = loss_fam_bin

            # 3. 16-Prototype Real-Camera Manifold Pull (Real -> <= tau) & Push (AI d_min -> >= tau + margin)
            real_pull = is_real * (d_train.pow(2) + 2.5 * F.relu(d_train - tau).pow(2))
            target_out = torch.clamp(tau.detach() + self.manifold_push_margin, min=0.46, max=1.20)
            # Push exact d_min so EVERY prototype stays well separated from synthetic embeddings!
            fake_push = is_fake * F.relu(target_out - d_min).pow(2)
            loss_manifold_cal = self._bce(manifold_logit, t_col)

            real_mask_1d = t_col.squeeze(1) < 0.5
            if int(real_mask_1d.sum().item()) >= 4:
                real_disp_var = d_min[real_mask_1d].var()
            else:
                real_disp_var = torch.zeros((), device=d_min.device, dtype=torch.float32)

            loss_sphere = (
                2.8 * torch.mean(real_pull + fake_push)
                + 0.35 * loss_manifold_cal
                + 1.4 * real_disp_var
            )

            # 4. Pair-Synchronized Internet-Laundered Twin Margin Ranking
            bsz = t_col.shape[0]
            if paired_batch and bsz >= 2 and bsz % 2 == 0:
                r_disc, a_disc = disc_logit[0::2], disc_logit[1::2]
                r_dist, a_dist = d_min[0::2], d_min[1::2]
                valid_pairs = ((t_col[0::2] < 0.5) & (t_col[1::2] >= 0.5)).to(torch.float32)
                if valid_pairs.sum() > 0:
                    p_logit = F.softplus(self.pair_margin_logit - (a_disc - r_disc)) * valid_pairs
                    p_dist = F.softplus((self.pair_margin_dist - (a_dist - r_dist)) * 6.0) * valid_pairs
                    denom = valid_pairs.sum() + 1e-6
                    loss_pair = 0.55 * (p_logit.sum() / denom) + 0.45 * (p_dist.sum() / denom)
                else:
                    loss_pair = torch.zeros((), device=d_min.device, dtype=torch.float32)
            else:
                loss_pair = torch.zeros((), device=d_min.device, dtype=torch.float32)

            # 5. Stream Gate Load-Balancing
            gate_mean = details["gate_weights"].to(torch.float32).mean(dim=0)
            target_gate = torch.full_like(gate_mean, 1.0 / len(STREAM_KEYS))
            loss_gate = F.mse_loss(gate_mean, target_gate)

            total = (
                0.30 * loss_main
                + 0.18 * loss_streams
                + 0.18 * loss_family
                + 0.19 * loss_sphere
                + 0.12 * loss_pair
                + 0.03 * loss_gate
            )
            return total


# Backward-compatible aliases
ArtifactNetV4Loss = ArtifactNetV5Loss
AsymmetricHypersphereLoss = ArtifactNetV5Loss


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.997) -> None:
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, ema_v in self.ema.state_dict().items():
            model_v = msd[k].detach()
            if ema_v.dtype.is_floating_point and k not in ("blend_alpha", "calib_temp", "calib_bias"):
                ema_v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)
            else:
                ema_v.copy_(model_v)


def compute_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> dict:
    y_true_i = np.asarray(y_true).astype(int)
    y_probs_f = np.asarray(y_probs).astype(np.float64)
    y_pred = (y_probs_f >= threshold).astype(int)

    real_mask = y_true_i == 0
    specificity = float((y_pred[real_mask] == 0).mean()) if np.any(real_mask) else 0.0

    return {
        "f1": float(f1_score(y_true_i, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true_i, y_pred)),
        "precision": float(precision_score(y_true_i, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_i, y_pred, zero_division=0)),
        "specificity": specificity,
        "roc_auc": float(roc_auc_score(y_true_i, y_probs_f)) if len(np.unique(y_true_i)) > 1 else 0.0,
    }


@torch.no_grad()
def evaluate_perturbation_suite(
    model: nn.Module,
    df: pd.DataFrame,
    data_dir: Path,
    device: torch.device,
    modes: tuple[str, ...] = (
        "whatsapp_1hop",
        "social_2hop_chain",
        "webp_cdn_q75",
        "screenshot_reencode",
        "phone_isp",
    ),
    samples_per_mode: int = 360,
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict[str, dict]:
    """
    Evaluates the model across deterministic internet-laundering, messenger, screenshot, and ISP transforms.
    """
    model.eval()
    half = max(10, samples_per_mode // 2)
    df0 = df[df["label"] == 0]
    df1 = df[df["label"] == 1]
    sub_df = pd.concat(
        [
            df0.sample(min(half, len(df0)), random_state=42),
            df1.sample(min(half, len(df1)), random_state=42),
        ],
        ignore_index=True,
    )

    results: dict[str, dict] = {}
    all_y, all_p = [], []
    dev_type = "cuda" if device.type == "cuda" else "cpu"
    use_amp = device.type == "cuda"

    for mode in modes:
        ds = SingleForensicDataset(sub_df, data_dir, img_size=256, perturbation=mode, return_global=True)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        ys, ps = [], []
        for xg, xr, xp, targets in loader:
            xg = xg.to(device, non_blocking=True)
            xr = xr.to(device, non_blocking=True)
            xp = xp.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_amp):
                logits, det = model(xr, xp, x_global=xg, return_details=True)
                l_f32 = logits.to(torch.float32)
                if det["stat_calibrated"]:
                    l_f32 = 0.90 * l_f32 + 0.10 * det["stat_logit"].to(torch.float32)
                probs = torch.sigmoid(l_f32).squeeze(1)
            ys.extend(targets.numpy())
            ps.extend(probs.cpu().numpy())

        m = compute_metrics(np.array(ys), np.array(ps))
        results[mode] = m
        all_y.extend(ys)
        all_p.extend(ps)

    results["aggregate_perturbed"] = compute_metrics(np.array(all_y), np.array(all_p))
    return results


def fit_post_training_calibration(
    model: nn.Module,
    val_df: pd.DataFrame,
    data_dir: Path,
    device: torch.device,
    val_ood_df: pd.DataFrame | None = None,
    samples_per_cond: int = 600,
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict:
    """
    Post-Training Unseen-Generator & Multi-Hop Internet Calibration (`ArtifactNet v5`):
    Jointly evaluates in-distribution AND strictly held-out unseen-generator (`val_ood_df`) samples across
    6 real-world delivery conditions (`clean`, `whatsapp_1hop`, `social_2hop_chain`, `webp_cdn_q75`,
    `screenshot_reencode`, `phone_isp`) to:
      1. Fit L2-regularized LogisticRegression weights for the 28 dimensionless + meso-scale physics ratios.
      2. Optimize `(blend_alpha, calib_temp, calib_bias)` with `blend_alpha >= 0.24` so the 16-Prototype
         Real-Camera Manifold stays active and `threshold = 0.50` achieves balanced Real Specificity & AI Recall
         on both clean and compressed/forwarded internet images!
    """
    model.eval()
    conditions = (
        "clean",
        "whatsapp_1hop",
        "social_2hop_chain",
        "webp_cdn_q75",
        "screenshot_reencode",
        "phone_isp",
    )

    # Combine 40% val_id + 60% val_ood (when available) so unseen generators & internet posts anchor the threshold!
    if val_ood_df is not None and len(val_ood_df) >= 40:
        n_id_half = max(20, int(samples_per_cond * 0.20))
        n_ood_half = max(20, int(samples_per_cond * 0.30))
        id0, id1 = val_df[val_df["label"] == 0], val_df[val_df["label"] == 1]
        od0, od1 = val_ood_df[val_ood_df["label"] == 0], val_ood_df[val_ood_df["label"] == 1]
        sub_df = pd.concat(
            [
                id0.sample(min(n_id_half, len(id0)), random_state=123),
                id1.sample(min(n_id_half, len(id1)), random_state=123),
                od0.sample(min(n_ood_half, len(od0)), random_state=456),
                od1.sample(min(n_ood_half, len(od1)), random_state=456),
            ],
            ignore_index=True,
        )
    else:
        half = max(20, samples_per_cond // 2)
        df0, df1 = val_df[val_df["label"] == 0], val_df[val_df["label"] == 1]
        sub_df = pd.concat(
            [
                df0.sample(min(half, len(df0)), random_state=123),
                df1.sample(min(half, len(df1)), random_state=123),
            ],
            ignore_index=True,
        )

    disc_list, d_real_list, phys_list, y_list = [], [], [], []
    dev_type = "cuda" if device.type == "cuda" else "cpu"
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for cond in conditions:
            ds = SingleForensicDataset(sub_df, data_dir, img_size=256, perturbation=cond, return_global=True)
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device.type == "cuda"),
            )
            for xg, xr, xp, targets in loader:
                xg = xg.to(device, non_blocking=True)
                xr = xr.to(device, non_blocking=True)
                xp = xp.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_amp):
                    _, det = model(xr, xp, x_global=xg, return_details=True)
                disc_list.append(det["disc_logit"].to(torch.float32).cpu())
                d_real_list.append(det["d_real"].to(torch.float32).cpu())
                phys_list.append(det["physics_vec"].to(torch.float32).cpu())
                y_list.append(targets.view(-1, 1).to(torch.float32).cpu())

    disc_all = torch.cat(disc_list, dim=0)
    d_real_all = torch.cat(d_real_list, dim=0)
    phys_all = torch.cat(phys_list, dim=0)
    y_all = torch.cat(y_list, dim=0)
    y_np = y_all.squeeze(1).numpy().astype(int)

    # Align 16-Prototype Real-Camera Manifold boundary (tau) & slope (m_scale) to empirical d_real distribution!
    with torch.no_grad():
        d_np = d_real_all.squeeze(1).numpy()
        r_med = float(np.percentile(d_np[y_np == 0], 68)) if np.any(y_np == 0) else 0.03
        a_med = float(np.percentile(d_np[y_np == 1], 38)) if np.any(y_np == 1) else 0.06
        emp_tau = float(np.clip(0.5 * (r_med + a_med), 0.012, 0.80))
        d_iqr = max(1e-3, float(np.percentile(d_np, 80) - np.percentile(d_np, 20)))
        disc_iqr = max(0.5, float(np.percentile(disc_all.numpy(), 80) - np.percentile(disc_all.numpy(), 20)))
        emp_scale = float(np.clip(disc_iqr / d_iqr, 2.5, 55.0))
        model.manifold_tau.fill_(emp_tau)
        model.manifold_log_scale.fill_(float(np.log(emp_scale)))
        man_all = emp_scale * (d_real_all - emp_tau)

    # 1. Fit L2-regularized Logistic Model on 28 dimensionless + meso-scale statistical descriptors
    clf = LogisticRegression(C=0.5, max_iter=1000, solver="lbfgs")
    clf.fit(phys_all.numpy(), y_np)
    stat_probs = clf.predict_proba(phys_all.numpy())[:, 1]
    stat_auc = float(roc_auc_score(y_np, stat_probs)) if len(np.unique(y_np)) > 1 else 0.5

    with torch.no_grad():
        model.stat_weights.copy_(torch.from_numpy(clf.coef_[0].astype(np.float32)).to(device))
        model.stat_bias.copy_(torch.from_numpy(clf.intercept_.astype(np.float32)).to(device))
        model.stat_calibrated.fill_(True)

    # 2. Grid-search blend_alpha in [0.24, 0.56] + optimize Platt/Temperature scaling (calib_temp, calib_bias)
    best_obj = float("inf")
    best_params = (0.34, 1.0, 0.0)

    for alpha_cand in np.linspace(0.24, 0.56, 9):
        raw_comb = (1.0 - float(alpha_cand)) * disc_all + float(alpha_cand) * man_all
        log_temp = torch.zeros(1, dtype=torch.float32, requires_grad=True)
        bias = torch.zeros(1, dtype=torch.float32, requires_grad=True)
        opt = torch.optim.LBFGS([log_temp, bias], lr=0.25, max_iter=70)

        def _closure():
            opt.zero_grad()
            temp = torch.clamp(torch.exp(log_temp), 0.35, 4.0)
            cal_logits = (raw_comb + bias) / temp
            loss = F.binary_cross_entropy_with_logits(cal_logits, y_all) + 1e-3 * (
                log_temp.pow(2) + bias.pow(2)
            )
            loss.backward()
            return loss

        opt.step(_closure)
        with torch.no_grad():
            t_val = float(torch.clamp(torch.exp(log_temp), 0.35, 4.0).item())
            b_val = float(torch.clamp(bias, -2.5, 2.5).item())
            cal_logits = (raw_comb + b_val) / t_val
            nll = float(F.binary_cross_entropy_with_logits(cal_logits, y_all).item())
            probs = torch.sigmoid(cal_logits).squeeze(1).numpy()
            m = compute_metrics(y_np, probs)
            balance_pen = abs(m["specificity"] - m["recall"]) * 0.22
            obj = nll - 0.30 * m["f1"] + balance_pen
            if obj < best_obj:
                best_obj = obj
                best_params = (float(alpha_cand), t_val, b_val)

    best_alpha, best_temp, best_bias = best_params
    with torch.no_grad():
        model.blend_alpha.fill_(best_alpha)
        model.calib_temp.fill_(best_temp)
        model.calib_bias.fill_(best_bias)

        final_cal_logits = ((1.0 - best_alpha) * disc_all + best_alpha * man_all + best_bias) / best_temp
        cal_probs = torch.sigmoid(final_cal_logits).squeeze(1).numpy()
        cal_metrics = compute_metrics(y_np, cal_probs)

    return {
        "blend_alpha": best_alpha,
        "calib_temp": best_temp,
        "calib_bias": best_bias,
        "stat_multi_cond_auc": stat_auc,
        "multi_cond_f1": cal_metrics["f1"],
        "multi_cond_specificity": cal_metrics["specificity"],
        "multi_cond_recall": cal_metrics["recall"],
        "multi_cond_auc": cal_metrics["roc_auc"],
    }
