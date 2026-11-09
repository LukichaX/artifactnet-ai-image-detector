import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.dataset import (
    CANONICAL_MAX_LONG_SIDE,
    canonicalize_scale,
    ensure_rgb_uint8,
    smash_and_reconstruct_rich_poor,
    strip_borders_and_canvas,
)
from src.model import FAMILY_NAMES, ArtifactNetV5


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-max(-25.0, min(25.0, x)))))


class ForensicCouncil:
    """
    ArtifactNet v5 Calibrated Multi-Scale & Internet-Resilient Forensic Inspector:
      - Evaluates each image across a 4-view Multi-Scale + Horizontal-Flip TTA pyramid
        (1024px native sensor/generator pitch + 640px meso/internet pitch) in a single batched GPU pass.
      - Fuses the 4 orthogonal neural streams, 12-Class Generator-Family Union head, 16-Prototype Real-Camera
        Manifold distance, and the data-calibrated 28-descriptor statistical physics model.
      - Reports Top-3 Suspected Generator Families and generates dual spatial XAI heatmaps.
    """

    def __init__(self, model: ArtifactNetV5, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.model.eval()

    @staticmethod
    def _build_views(
        clean_img: np.ndarray, img_size: int = 256, use_tta: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c_primary = canonicalize_scale(clean_img, max_long_side=CANONICAL_MAX_LONG_SIDE)
        xg0 = cv2.resize(c_primary, (img_size, img_size), interpolation=cv2.INTER_AREA)
        xr0, xp0 = smash_and_reconstruct_rich_poor(c_primary, out_size=img_size, patch_size=32, is_train=False)

        if not use_tta:
            return xg0[None], xr0[None], xp0[None]

        c_primary_flip = cv2.flip(c_primary, 1)
        xg1 = cv2.resize(c_primary_flip, (img_size, img_size), interpolation=cv2.INTER_AREA)
        xr1, xp1 = smash_and_reconstruct_rich_poor(c_primary_flip, out_size=img_size, patch_size=32, is_train=False)

        c_sec = canonicalize_scale(clean_img, target_long_side=640)
        xg2 = cv2.resize(c_sec, (img_size, img_size), interpolation=cv2.INTER_AREA)
        xr2, xp2 = smash_and_reconstruct_rich_poor(c_sec, out_size=img_size, patch_size=32, is_train=False)

        c_sec_flip = cv2.flip(c_sec, 1)
        xg3 = cv2.resize(c_sec_flip, (img_size, img_size), interpolation=cv2.INTER_AREA)
        xr3, xp3 = smash_and_reconstruct_rich_poor(c_sec_flip, out_size=img_size, patch_size=32, is_train=False)

        xg = np.stack([xg0, xg1, xg2, xg3], axis=0)
        xr = np.stack([xr0, xr1, xr2, xr3], axis=0)
        xp = np.stack([xp0, xp1, xp2, xp3], axis=0)
        return xg, xr, xp

    @torch.no_grad()
    def predict_prob(self, raw_img_rgb: np.ndarray, use_tta: bool = True) -> float:
        img = ensure_rgb_uint8(raw_img_rgb)
        clean_img, _, _ = strip_borders_and_canvas(img)
        xg_np, xr_np, xp_np = self._build_views(clean_img, img_size=256, use_tta=use_tta)

        xg = torch.from_numpy(np.ascontiguousarray(xg_np.transpose(0, 3, 1, 2))).to(self.device)
        xr = torch.from_numpy(np.ascontiguousarray(xr_np.transpose(0, 3, 1, 2))).to(self.device)
        xp = torch.from_numpy(np.ascontiguousarray(xp_np.transpose(0, 3, 1, 2))).to(self.device)

        dev_type = "cuda" if self.device.type == "cuda" else "cpu"
        use_amp = self.device.type == "cuda"
        with torch.amp.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_amp):
            final_logit, det = self.model(xr, xp, x_global=xg, return_details=True)

        mean_logit = float(final_logit.to(torch.float32).mean().item())
        if det["stat_calibrated"]:
            stat_l = float(det["stat_logit"].to(torch.float32).mean().item())
            mean_logit = 0.90 * mean_logit + 0.10 * stat_l

        return _sigmoid(mean_logit)

    @torch.no_grad()
    def _build_xai_overlays(
        self,
        clean_img: np.ndarray,
        xr0: np.ndarray,
        anomaly_8x8: torch.Tensor,
        physics_maps: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        disp_img = canonicalize_scale(clean_img, max_long_side=768)
        dh, dw = disp_img.shape[:2]

        patch_map = F.interpolate(
            anomaly_8x8[0:1], size=(dh, dw), mode="bicubic", align_corners=False
        )[0, 0].cpu().numpy()
        phys_comb = physics_maps[0:1, 1:2].abs() * 0.5 + physics_maps[0:1, 4:5].abs() * 0.5
        phys_up = F.interpolate(phys_comb, size=(dh, dw), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

        p_lo, p_hi = np.percentile(patch_map, 10), np.percentile(patch_map, 98)
        patch_norm = np.clip((patch_map - p_lo) / (p_hi - p_lo + 1e-6), 0.0, 1.0)
        phys_norm = np.clip(phys_up / (np.percentile(phys_up, 98) + 1e-6), 0.0, 1.0)
        scene_heat = np.clip(0.65 * patch_norm + 0.35 * phys_norm, 0.0, 1.0)

        scene_cmap = cv2.cvtColor(
            cv2.applyColorMap((scene_heat * 255).astype(np.uint8), cv2.COLORMAP_TURBO),
            cv2.COLOR_BGR2RGB,
        )
        scene_overlay = cv2.addWeighted(disp_img, 0.55, scene_cmap, 0.45, 0)

        xr_t = (
            torch.from_numpy(np.ascontiguousarray(xr0.transpose(2, 0, 1)))
            .unsqueeze(0)
            .to(self.device, dtype=torch.float32)
            / 255.0
        )
        lpd_abs = self.model.stream1_micro.frontend.lpd(xr_t)[0, 3:6].mean(dim=0)
        bayar_abs = self.model.stream1_micro.frontend.bayar(xr_t)[0].abs().mean(dim=0)
        micro_raw = (0.55 * lpd_abs + 0.45 * bayar_abs).cpu().numpy()
        micro_norm = np.clip(micro_raw / (np.percentile(micro_raw, 98) + 1e-6), 0.0, 1.0)
        micro_cmap = cv2.cvtColor(
            cv2.applyColorMap((micro_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO),
            cv2.COLOR_BGR2RGB,
        )
        micro_overlay = cv2.addWeighted(xr0, 0.45, micro_cmap, 0.55, 0)
        return scene_overlay, micro_overlay

    @torch.no_grad()
    def analyze(self, raw_img_rgb: np.ndarray, use_tta: bool = True) -> dict:
        img = ensure_rgb_uint8(raw_img_rgb)
        orig_h, orig_w = img.shape[:2]
        clean_img, canvas_ratio, is_monochrome = strip_borders_and_canvas(img)

        xg_np, xr_np, xp_np = self._build_views(clean_img, img_size=256, use_tta=use_tta)
        xg = torch.from_numpy(np.ascontiguousarray(xg_np.transpose(0, 3, 1, 2))).to(self.device)
        xr = torch.from_numpy(np.ascontiguousarray(xr_np.transpose(0, 3, 1, 2))).to(self.device)
        xp = torch.from_numpy(np.ascontiguousarray(xp_np.transpose(0, 3, 1, 2))).to(self.device)

        dev_type = "cuda" if self.device.type == "cuda" else "cpu"
        use_amp = self.device.type == "cuda"
        with torch.amp.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_amp):
            final_logit, det = self.model(
                xr, xp, x_global=xg, return_details=True, return_maps=True
            )

        view_probs = torch.sigmoid(final_logit.to(torch.float32)).squeeze(1).cpu().numpy()
        mean_neural_logit = float(final_logit.to(torch.float32).mean().item())

        stat_cal = det["stat_calibrated"]
        stat_logit_mean = float(det["stat_logit"].to(torch.float32).mean().item())
        if stat_cal:
            ensemble_logit = 0.90 * mean_neural_logit + 0.10 * stat_logit_mean
        else:
            ensemble_logit = mean_neural_logit

        final_ai_prob = float(np.clip(_sigmoid(ensemble_logit), 0.001, 0.999))

        temp = float(self.model.calib_temp.item())
        p_s1 = _sigmoid(float(det["logit1_micro"].to(torch.float32).mean().item()) / temp)
        p_s2 = _sigmoid(float(det["logit2_wavelet"].to(torch.float32).mean().item()) / temp)
        p_s3 = _sigmoid(float(det["logit3_global"].to(torch.float32).mean().item()) / temp)
        p_s4 = _sigmoid(float(det["logit4_gram_phys"].to(torch.float32).mean().item()) / temp)
        p_fam_union = _sigmoid(float(det["family_ai_logit"].to(torch.float32).mean().item()) / temp)
        p_manifold = _sigmoid(float(det["manifold_logit"].to(torch.float32).mean().item()) / temp)
        p_disc = _sigmoid(float(det["disc_logit"].to(torch.float32).mean().item()) / temp)

        # Top-3 Suspected Synthetic Generator Families (conditional distribution across AI families 1..11)
        fam_logits_mean = det["family_logits"].to(torch.float32).mean(dim=0).cpu()
        synth_probs = F.softmax(fam_logits_mean[1:], dim=0).numpy()
        top_idx = np.argsort(synth_probs)[::-1][:3]
        top_families = [
            (FAMILY_NAMES[int(idx) + 1], float(synth_probs[int(idx)])) for idx in top_idx
        ]

        mean_d_real = float(det["d_real"].to(torch.float32).mean().item())
        tau = float(det["manifold_tau"].to(torch.float32).item())
        gate_w = det["gate_weights"].to(torch.float32).mean(dim=0).cpu().numpy()

        phys_mean = det["physics_vec"].mean(dim=0).cpu()
        w_stat = self.model.stat_weights.detach().cpu()
        b_stat = float(self.model.stat_bias.detach().cpu().item())

        if stat_cal and phys_mean.numel() == w_stat.numel():
            contrib = phys_mean * w_stat
            p_stat_total = _sigmoid(float(contrib.sum().item() + b_stat))
            p_phys_rp = _sigmoid(float(contrib[3:6].sum().item() * 2.0 + b_stat * 0.20))
            p_phys_spec = _sigmoid(float((contrib[6:11].sum() + contrib[22]).item() * 1.8 + b_stat * 0.20))
            p_phys_opt = _sigmoid(float(contrib[11:17].sum().item() * 1.8 + b_stat * 0.20))
            p_phys_wav = _sigmoid(float((contrib[0:3].sum() + contrib[17:22].sum() + contrib[23]).item() * 1.5 + b_stat * 0.20))
            p_phys_meso = _sigmoid(float(contrib[24:28].sum().item() * 2.0 + b_stat * 0.20))
        else:
            p_stat_total = p_s4
            p_phys_rp = p_s1
            p_phys_spec = p_s2
            p_phys_opt = p_s3
            p_phys_wav = p_s2
            p_phys_meso = p_s3

        experts = [
            ("Stream 1: Rich-vs-Poor Micro-Residuals (NPR/LPD/SRM/Bayar)", p_s1, float(gate_w[0])),
            ("Stream 2: Multi-Scale Wavelet & 2D-FFT Phase Coherence", p_s2, float(gate_w[1])),
            ("Stream 3: Meso-Structural & 64-Patch Self-Consistency ViT", p_s3, float(gate_w[2])),
            ("Stream 4: Differential Gram Covariance & 28-Physics MLP", p_s4, float(gate_w[3])),
            ("12-Family Multi-Generator Union Head", p_fam_union, 0.30),
            ("Real-Camera Manifold Anomaly (16-Prototype Hypersphere)", p_manifold, float(self.model.blend_alpha.item())),
            ("Calibrated Statistical Physics: Intra-Image Noise Floor", p_phys_rp, 0.08),
            ("Calibrated Statistical Physics: Spectral Slope & Anisotropy", p_phys_spec, 0.08),
            ("Calibrated Statistical Physics: Poisson Shot & Chroma-Luma", p_phys_opt, 0.08),
            ("Calibrated Statistical Physics: Wavelet & Heavy-Tail NSS", p_phys_wav, 0.08),
            ("Calibrated Statistical Physics: Meso-Scale & Compression-Survivable", p_phys_meso, 0.08),
        ]

        flagged_count = sum(1 for _, sc, _ in experts if sc >= 0.50)
        tta_agreement = float(np.mean((view_probs >= 0.5) == (final_ai_prob >= 0.5)))

        scene_overlay, micro_overlay = self._build_xai_overlays(
            clean_img=clean_img,
            xr0=xr_np[0],
            anomaly_8x8=det["anomaly_8x8"],
            physics_maps=det["physics_maps"],
        )

        return {
            "ai_prob": final_ai_prob,
            "real_prob": 1.0 - final_ai_prob,
            "disc_prob": p_disc,
            "manifold_prob": p_manifold,
            "family_union_prob": p_fam_union,
            "top_families": top_families,
            "stat_prob": p_stat_total,
            "stat_calibrated": stat_cal,
            "tta_view_probs": [float(v) for v in view_probs],
            "tta_agreement": tta_agreement,
            "flagged_count": flagged_count,
            "total_experts": len(experts),
            "experts": experts,
            "scene_overlay": scene_overlay,
            "micro_overlay": micro_overlay,
            "xr_vis": xr_np[0],
            "xp_vis": xp_np[0],
            "orig_size": (orig_w, orig_h),
            "is_monochrome": is_monochrome,
            "canvas_ratio": canvas_ratio,
            "d_real": mean_d_real,
            "manifold_tau": tau,
        }


ForensicInspectorV4 = ForensicCouncil
ForensicInspectorV5 = ForensicCouncil
