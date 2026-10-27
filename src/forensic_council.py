import math
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.dataset import smash_and_reconstruct_rich_poor
from src.model import ArtifactNetV2


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x)))))


def strip_borders_and_canvas(img_rgb: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """
    ავტომატურად პოულობს და აჭრის თეთრ/შავ/ერთფეროვან ჩარჩოებსა და კანვასს,
    რათა კოლაჟებმა ან სქრინშოტების კიდეებმა არ მოატყუოს FFT და Hypersphere.
    ასევე ზუსტად ადგენს შავ-თეთრ / მონოქრომულ / სეპია რეჟიმს.
    """
    h, w, _ = img_rgb.shape
    img_f = img_rgb.astype(np.float32)

    # 1. შავ-თეთრი ან სეპია/დასკანერებული მონოქრომული ფოტოს დეტექცია
    rg_diff = np.abs(img_f[:, :, 0] - img_f[:, :, 1])
    gb_diff = np.abs(img_f[:, :, 1] - img_f[:, :, 2])
    rb_diff = np.abs(img_f[:, :, 0] - img_f[:, :, 2])
    mean_chroma_spread = float(np.mean(rg_diff + gb_diff + rb_diff) / 3.0)

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat_90 = float(np.percentile(hsv[:, :, 1], 90))
    is_monochrome = (mean_chroma_spread < 11.0) or (sat_90 < 28.0)

    # 2. კანვასის / ჩარჩოს წილის დათვლა და სუფთა შიგთავსის ამოჭრა
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.abs(gx) + np.abs(gy)

    # პიქსელი ითვლება კანვასად, თუ ძალიან თეთრია (>242) ან ძალიან შავი (<12) და გრადიენტი ნულია
    is_canvas_px = ((gray > 240) | (gray < 14)) & (grad < 8.0)
    canvas_ratio = float(np.mean(is_canvas_px))

    # ვპოულობთ აქტიურ კონტენტს კიდეებიდან
    row_active = np.mean(~is_canvas_px, axis=1) > 0.15
    col_active = np.mean(~is_canvas_px, axis=0) > 0.15

    y_idx = np.where(row_active)[0]
    x_idx = np.where(col_active)[0]

    if len(y_idx) > 64 and len(x_idx) > 64:
        y1, y2 = int(y_idx[0]), int(y_idx[-1]) + 1
        x1, x2 = int(x_idx[0]), int(x_idx[-1]) + 1
        if (y2 - y1) >= 128 and (x2 - x1) >= 128:
            cropped = img_rgb[y1:y2, x1:x2]
        else:
            cropped = img_rgb
    else:
        cropped = img_rgb

    return cropped, canvas_ratio, is_monochrome


def filter_non_canvas_patches(img_rgb: np.ndarray, out_size: int = 256, patch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """
    PatchCraft-ის გაუმჯობესებული ვერსია ინფერენსისთვის:
    უგულებელყოფს ცარიელ თეთრ/შავ კანვასის კუბიკებს (რათა კოლაჟის თეთრი ფონი
    არ მოხვდეს Poor ტექსტურაში და არ გამოიწვიოს ყალბი AI განგაში).
    """
    h, w, c = img_rgb.shape
    if h < out_size or w < out_size:
        pad_h = max(0, out_size - h)
        pad_w = max(0, out_size - w)
        img_rgb = cv2.copyMakeBorder(img_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        h, w, c = img_rgb.shape

    grid_n = out_size // patch_size
    needed = grid_n * grid_n
    nh, nw = h // patch_size, w // patch_size
    cropped = img_rgb[: nh * patch_size, : nw * patch_size]
    patches = cropped.reshape(nh, patch_size, nw, patch_size, c).swapaxes(1, 2).reshape(-1, patch_size, patch_size, c)

    p_f = patches.astype(np.float32)
    dh = np.abs(p_f[:, :, 1:, :] - p_f[:, :, :-1, :]).mean(axis=(1, 2, 3))
    dv = np.abs(p_f[:, 1:, :, :] - p_f[:, :-1, :, :]).mean(axis=(1, 2, 3))
    dd1 = np.abs(p_f[:, 1:, 1:, :] - p_f[:, :-1, :-1, :]).mean(axis=(1, 2, 3))
    dd2 = np.abs(p_f[:, 1:, :-1, :] - p_f[:, :-1, 1:, :]).mean(axis=(1, 2, 3))
    l_div = dh + dv + dd1 + dd2

    # ვფილტრავთ მკვდარ/ცარიელ კანვასის კვადრატებს (სადაც l_div < 1.2)
    valid_mask = l_div > 1.2
    if np.sum(valid_mask) >= 12:
        valid_patches = patches[valid_mask]
        valid_ldiv = l_div[valid_mask]
    else:
        valid_patches = patches
        valid_ldiv = l_div

    order = np.argsort(valid_ldiv)[::-1]
    pool_k = max(4, len(order) // 3)
    rich_pool = order[:pool_k]
    poor_pool = order[-pool_k:]

    rich_chosen = np.resize(rich_pool, needed)
    poor_chosen = np.resize(poor_pool, needed)

    x_rich = valid_patches[rich_chosen].reshape(grid_n, grid_n, patch_size, patch_size, c).swapaxes(1, 2).reshape(out_size, out_size, c)
    x_poor = valid_patches[poor_chosen].reshape(grid_n, grid_n, patch_size, patch_size, c).swapaxes(1, 2).reshape(out_size, out_size, c)
    return x_rich, x_poor


class ForensicCouncil:
    """
    ArtifactNet v3.5: 15-Expert Forensic Council (100% From Scratch).
    აერთიანებს 3 ღრმა ნეირონულ ექსპერტს (ArtifactNetV2) და 12 ფიზიკურ/მათემატიკურ ექსპერტს
    კვორუმზე დაფუძნებული გადაწყვეტილების სისტემით (Quorum Consensus Voting).
    """
    def __init__(self, model: ArtifactNetV2, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def _run_deep_experts(
        self, img_rgb: np.ndarray, is_monochrome: bool, canvas_ratio: float
    ) -> tuple[float, float, float, np.ndarray, np.ndarray, float, float]:
        xr1, xp1 = filter_non_canvas_patches(img_rgb, out_size=256, patch_size=32)
        img_flip = cv2.flip(img_rgb, 1)
        xr2, xp2 = filter_non_canvas_patches(img_flip, out_size=256, patch_size=32)

        xr = torch.from_numpy(np.ascontiguousarray(np.stack([xr1, xr2]).transpose(0, 3, 1, 2))).to(self.device)
        xp = torch.from_numpy(np.ascontiguousarray(np.stack([xp1, xp2]).transpose(0, 3, 1, 2))).to(self.device)

        x_rich_f = xr.to(torch.float32) / 255.0
        x_poor_f = xp.to(torch.float32) / 255.0

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            b = x_rich_f.shape[0]
            x_both = torch.cat([x_rich_f, x_poor_f], dim=0)
            f0 = self.model.frontend(x_both)
            f1 = self.model.stage1(f0)
            f2 = self.model.stage2(f1)

            f2_rich, f2_poor = f2[:b], f2[b:]
            c_rich = self.model.contrast_head(f2_rich)
            c_poor = self.model.contrast_head(f2_poor)
            contrast_vec = F.adaptive_avg_pool2d(c_rich - c_poor, 1).flatten(1)
            spectral_vec = self.model.spectral_branch(f2_rich)

            f3 = self.model.stage3(f2_rich)
            f4 = self.model.stage4(f3)
            avg_p = F.adaptive_avg_pool2d(f4, 1).flatten(1)
            max_p = F.adaptive_max_pool2d(f4, 1).flatten(1)
            spatial_vec = torch.cat([avg_p, max_p], dim=1)

            combined = torch.cat([spatial_vec, contrast_vec, spectral_vec], dim=1)
            z = self.model.embedding_proj(combined)

            center = self.model.real_center.to(z.dtype).unsqueeze(0)
            dist_sq = torch.mean((z - center).pow(2), dim=1, keepdim=True)
            radius_sq = F.softplus(self.model.log_radius_sq).to(z.dtype)

            linear_logit = self.model.linear_head(z)

            # 1. Expert #1: Deep Hypersphere Anomaly
            ratio = float((dist_sq.mean() / (radius_sq + 1e-6)).item())
            if is_monochrome or canvas_ratio > 0.15:
                # შავ-თეთრ ან კოლაჟირებულ ფოტოებზე ჰიპერსფეროს რადიუსს ვარბილებთ
                ratio *= 0.45
            prob_sphere = _sigmoid((ratio - 1.35) * 2.2)

            # 2. Expert #2: Deep PatchCraft & FreqNet Energy
            contrast_energy = float(contrast_vec.float().norm(dim=1).mean().item())
            spec_energy = float(spectral_vec.float().norm(dim=1).mean().item())
            cf_score = (contrast_energy / 8.5) + (spec_energy / 11.0) - 1.55
            prob_patch_freq = _sigmoid(cf_score * 2.0)

            # 3. Expert #3: Deep Learned Pattern (Calibrated Linear Logit)
            raw_lin = float(linear_logit.float().mean().item())
            # ვამცირებთ ტრენინგის დროინდელ ჭარბ AI-bias-ს ტემპერატურული დაწევით
            prob_linear = _sigmoid((raw_lin - 0.65) * 0.55)

        return (
            prob_sphere,
            prob_patch_freq,
            prob_linear,
            xr1,
            xp1,
            float(dist_sq.mean().item()),
            float(radius_sq.item()),
        )

    def _expert_bokeh_dof(self, img_rgb: np.ndarray) -> float:
        """Expert #4: Bokeh & Optical Depth-of-Field Physics."""
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        noise_res = gray - cv2.GaussianBlur(gray, (3, 3), 0)

        h, w = gray.shape
        gh, gw = max(2, h // 64), max(2, w // 64)
        sharp_vars, noise_stds = [], []
        for r in range(gh):
            for c in range(gw):
                y1, y2 = r * 64, min(h, (r + 1) * 64)
                x1, x2 = c * 64, min(w, (c + 1) * 64)
                sharp_vars.append(float(np.var(lap[y1:y2, x1:x2])))
                noise_stds.append(float(np.std(noise_res[y1:y2, x1:x2])))

        sharp_arr = np.array(sharp_vars)
        noise_arr = np.array(noise_stds)
        if len(sharp_arr) < 4:
            return 0.35

        q75, q25 = np.percentile(sharp_arr, 75), np.percentile(sharp_arr, 25)
        fg_noise = float(np.mean(noise_arr[sharp_arr >= q75]) + 1e-6)
        bg_noise = float(np.mean(noise_arr[sharp_arr <= q25]) + 1e-6)

        # AI პორტრეტებში უკანა ფონი (Bokeh) სრულიად გაუთოებულია (bg_noise << fg_noise)
        dof_drop = fg_noise / bg_noise
        return _sigmoid((dof_drop - 3.1) * 0.95)

    def _expert_smooth_plasticity(self, img_rgb: np.ndarray, is_monochrome: bool) -> float:
        """Expert #5: Smooth-Surface Plasticity & Fake-Grain (AI Skin / Airbrushed Surfaces)."""
        if is_monochrome:
            return 0.35
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        blur = cv2.bilateralFilter(gray, d=5, sigmaColor=20, sigmaSpace=20)
        micro_res = np.abs(gray - blur)

        # ვპოულობთ საშუალო ტონალობის (კანის/ზედაპირის) გლუვ უბნებს
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.hypot(gx, gy)

        smooth_mask = (grad < 18.0) & (gray > 35.0) & (gray < 225.0)
        if np.sum(smooth_mask) < 500:
            return 0.40

        smooth_noise = float(np.percentile(micro_res[smooth_mask], 50))
        # თუ გლუვ ზედაპირზე მიკრო-ხმაური < 0.85-ია, ზედაპირი ხელოვნურად პლასტმასისებრია
        return _sigmoid((1.05 - smooth_noise) * 2.4)

    def _expert_color_gamut(self, img_rgb: np.ndarray, is_monochrome: bool) -> float:
        """Expert #6: Color Gamut & Synthetic Hue-Saturation Coupling (დაკალიბრებული)."""
        if is_monochrome:
            return 0.20
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        sat = hsv[:, :, 1] / 255.0
        val = hsv[:, :, 2] / 255.0

        # მხოლოდ ექსტრემალური ნეონური/ციფრული გაჯერება (და არა ჩვეულებრივი მზიანი ფოტო!)
        extreme_sat_ratio = float(np.mean((sat > 0.82) & (val > 0.25) & (val < 0.92)))
        sat_p98 = float(np.percentile(sat, 98))

        score = (extreme_sat_ratio - 0.08) * 8.0 + (sat_p98 - 0.88) * 3.0
        return _sigmoid(score)

    def _expert_mscn_statistics(self, img_rgb: np.ndarray) -> float:
        """Expert #7: Natural Scene Statistics (MSCN Gaussianity / Kurtosis)."""
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        mu = cv2.GaussianBlur(gray, (7, 7), 1.166)
        mu_sq = mu * mu
        sigma = np.sqrt(np.maximum(cv2.GaussianBlur(gray * gray, (7, 7), 1.166) - mu_sq, 0.0))
        mscn = (gray - mu) / (sigma + 1.0)

        var = float(np.var(mscn))
        kurt = float(np.mean((mscn - np.mean(mscn)) ** 4) / (var ** 2 + 1e-6))
        # ბუნებრივ ფოტოებს აქვთ Kurtosis ~ 2.2 - 4.8. AI სურათებს ხშირად > 5.5 ან < 1.8
        dev = max(0.0, kurt - 4.8) + max(0.0, 2.0 - kurt)
        return _sigmoid((dev - 0.75) * 1.1)

    def _expert_gradient_heavytail(self, img_rgb: np.ndarray, canvas_ratio: float) -> float:
        """Expert #8: Gradient Heavy-Tail & Hyper-Sharp Edge Profile."""
        if canvas_ratio > 0.15:
            return 0.35
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        dx = np.abs(gray[:, 1:] - gray[:, :-1])
        dy = np.abs(gray[1:, :] - gray[:-1, :])
        g = np.concatenate([dx.ravel(), dy.ravel()])

        p99 = float(np.percentile(g, 99))
        p50 = float(np.percentile(g, 50) + 1e-5)
        ratio = p99 / p50
        # AI გენერატორი ქმნის მკვეთრ კონტურებს სრულიად გლუვი გარემოს ფონზე (მაღალი p99/p50)
        return _sigmoid((ratio - 26.0) * 0.12)

    def _expert_fft_radial_decay(self, img_rgb: np.ndarray, canvas_ratio: float) -> float:
        """Expert #9: 2D-FFT Radial Power Law Decay (1/f^alpha)."""
        if canvas_ratio > 0.15:
            return 0.35
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
        gray = gray - np.mean(gray)
        win = np.outer(np.hanning(512), np.hanning(512))
        fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(gray * win))) ** 2 + 1e-8
        log_p = np.log10(fft_mag)

        y, x = np.indices((512, 512))
        r = np.hypot(x - 256, y - 256).astype(np.int32)
        radii = np.arange(15, 220)
        radial_prof = np.array([np.mean(log_p[r == rad]) for rad in radii])

        log_f = np.log10(radii)
        slope, _ = np.polyfit(log_f, radial_prof, 1)
        # ბუნებრივი ლინზის სპექტრის დახრილობაა დაახლოებით -2.0-დან -3.1-მდე
        dev = max(0.0, abs(slope + 2.55) - 0.50)
        return _sigmoid((dev - 0.25) * 3.2)

    def _expert_upsampling_residual(self, img_rgb: np.ndarray) -> float:
        """Expert #10: Multi-Scale Upsampling Periodic Residual (VAE/GAN Grid)."""
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        h, w = gray.shape
        h2, w2 = (h // 2) * 2, (w // 2) * 2
        g = gray[:h2, :w2]

        # 2x2 პერიოდული ფაზების დისბალანსი (Up-sampling checkerboard)
        p00 = g[0::2, 0::2]
        p01 = g[0::2, 1::2]
        p10 = g[1::2, 0::2]
        p11 = g[1::2, 1::2]

        d_horiz = np.mean(np.abs(p00 - p01))
        d_diag = np.mean(np.abs(p00 - p11))
        anisotropy = abs(d_diag - d_horiz * 1.414) / (d_horiz + 1.0)
        return _sigmoid((anisotropy - 0.19) * 9.0)

    def _expert_bayer_cfa_demosaicing(self, img_rgb: np.ndarray, is_monochrome: bool) -> float:
        """
        ახალი Expert #11: Bayer CFA Demosaicing Periodicity.
        ამოწმებს კამერის ფიზიკური RGGB სენსორის დემოზაიკინგის კორელაციას.
        """
        if is_monochrome:
            return 0.35
        img_f = img_rgb.astype(np.float32)
        g = img_f[:, :, 1]
        r = img_f[:, :, 0]
        b = img_f[:, :, 2]

        # Green-Red და Green-Blue მაღალსიხშირული თანხვედრა
        lap_k = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32) / 4.0
        hp_g = cv2.filter2D(g, cv2.CV_32F, lap_k)
        hp_r = cv2.filter2D(r, cv2.CV_32F, lap_k)
        hp_b = cv2.filter2D(b, cv2.CV_32F, lap_k)

        diff_rg = float(np.std(hp_g - hp_r))
        diff_gb = float(np.std(hp_g - hp_b))
        chroma_noise = 0.5 * (diff_rg + diff_gb)

        # AI დეკოდერებს აქვთ არაბუნებრივად დაბალი ქრომატული დემოზაიკინგის მიკრო-დისპერსია
        return _sigmoid((0.95 - chroma_noise) * 2.1)

    def _expert_photon_shot_noise(self, img_rgb: np.ndarray) -> float:
        """
        ახალი Expert #12: Photon Shot-Noise Law (Poisson Heteroscedasticity).
        რეალურ სენსორში ხმაური იზრდება სინათლის ინტენსივობასთან ერთად.
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        smooth = cv2.GaussianBlur(gray, (5, 5), 1.0)
        res = np.abs(gray - smooth)

        dark_mask = (smooth > 15) & (smooth < 75)
        bright_mask = (smooth > 140) & (smooth < 235)

        if np.sum(dark_mask) < 300 or np.sum(bright_mask) < 300:
            return 0.40

        dark_noise = float(np.mean(res[dark_mask]) + 1e-5)
        bright_noise = float(np.mean(res[bright_mask]) + 1e-5)
        ratio = bright_noise / dark_noise

        # AI სურათებში ხშირად ნათელი უბნები ზედმეტად გადაპრიალებულია ან ჩრდილი მკვდარია
        anomaly = max(0.0, 0.85 - ratio) + max(0.0, ratio - 3.8) * 0.4
        return _sigmoid((anomaly - 0.22) * 4.5)

    def _expert_chromatic_aberration(self, img_rgb: np.ndarray, is_monochrome: float) -> float:
        """
        ახალი Expert #13: Optical Lens Lateral Chromatic Aberration.
        ამოწმებს სურათის კიდეებში R და B არხების ბუნებრივ ოპტიკურ დისპერსიას.
        """
        if is_monochrome:
            return 0.35
        h, w, _ = img_rgb.shape
        r = img_rgb[:, :, 0].astype(np.float32)
        b = img_rgb[:, :, 2].astype(np.float32)

        gx_r = cv2.Sobel(r, cv2.CV_32F, 1, 0, ksize=3)
        gy_r = cv2.Sobel(r, cv2.CV_32F, 0, 1, ksize=3)
        gx_b = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
        gy_b = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)

        grad_r = np.hypot(gx_r, gy_r)
        grad_b = np.hypot(gx_b, gy_b)

        # კიდეების 20%-იანი ზონა vs ცენტრი
        cy1, cy2 = int(h * 0.25), int(h * 0.75)
        cx1, cx2 = int(w * 0.25), int(w * 0.75)
        diff_rb = np.abs(grad_r - grad_b)

        center_aberr = float(np.mean(diff_rb[cy1:cy2, cx1:cx2]) + 1e-5)
        border_mask = np.ones((h, w), dtype=bool)
        border_mask[cy1:cy2, cx1:cx2] = False
        border_aberr = float(np.mean(diff_rb[border_mask]) + 1e-5)

        # თუ კიდეებზე R-B გაშლა საერთოდ არ არსებობს და მთლიანი სურათი სტერილურია
        sterility = 1.0 / (border_aberr + center_aberr + 0.1)
        return _sigmoid((sterility - 0.18) * 6.0)

    def _expert_jpeg_block_grid(self, img_rgb: np.ndarray) -> float:
        """
        ახალი Expert #14: JPEG 8x8 Block Grid Artifact vs Synthetic Smoothness.
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        h, w = gray.shape
        if h < 64 or w < 64:
            return 0.40

        diff_h = np.abs(gray[:, 1:] - gray[:, :-1])
        # 8-პიქსელიანი ბადის საზღვრები vs შიდა პიქსელები
        grid_cols = diff_h[:, 7::8]
        non_grid_cols = np.delete(diff_h, np.arange(7, diff_h.shape[1], 8), axis=1)

        blockiness = float(np.mean(grid_cols)) / (float(np.mean(non_grid_cols)) + 1e-5)
        # თუ სურათს აქვს მკაფიო ბუნებრივი კამერის/JPEG 8x8 ბლოკურობა (>1.08), ის უფრო რეალურია
        return _sigmoid((1.01 - blockiness) * 8.0)

    def _expert_micro_texture_entropy(self, img_rgb: np.ndarray) -> float:
        """
        ახალი Expert #15: Micro-Texture Local Residual Entropy.
        იჭერს AI-ის ხელოვნურად დახატულ განმეორებად ტექსტურას (თმა, ფოთლები, ქსოვილი).
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        med = cv2.medianBlur(img_rgb[:, :, 1], 3).astype(np.float32)
        res = np.clip(np.round(gray - med) + 16, 0, 32).astype(np.int32)

        hist = np.bincount(res.ravel(), minlength=33).astype(np.float32)
        hist /= np.sum(hist) + 1e-8
        entropy = -float(np.sum(hist * np.log2(hist + 1e-8)))

        # რეალურ ფოტოში მიკრო-ნარჩენის ენტროპია მაღალია (> 2.35), AI-ში კი შეკუმშულია
        return _sigmoid((2.25 - entropy) * 1.9)

    def analyze(self, raw_img_rgb: np.ndarray) -> dict:
        orig_h, orig_w = raw_img_rgb.shape[:2]

        # 1. ვჭრით გარე თეთრ/შავ კანვასს და ვადგენთ შავ-თეთრ რეჟიმს
        clean_img, canvas_ratio, is_monochrome = strip_borders_and_canvas(raw_img_rgb)

        # 2. ვუშვებთ 3 ღრმა ნეირონულ ექსპერტს
        p_sphere, p_patch_freq, p_linear, xr_vis, xp_vis, mean_dist, mean_rad = self._run_deep_experts(
            clean_img, is_monochrome=is_monochrome, canvas_ratio=canvas_ratio
        )

        # 3. ვუშვებთ 12 ფიზიკურ და სტატისტიკურ ექსპერტს
        p_bokeh = self._expert_bokeh_dof(clean_img)
        p_plastic = self._expert_smooth_plasticity(clean_img, is_monochrome)
        p_color = self._expert_color_gamut(clean_img, is_monochrome)
        p_mscn = self._expert_mscn_statistics(clean_img)
        p_grad = self._expert_gradient_heavytail(clean_img, canvas_ratio)
        p_fft = self._expert_fft_radial_decay(clean_img, canvas_ratio)
        p_upsample = self._expert_upsampling_residual(clean_img)
        p_cfa = self._expert_bayer_cfa_demosaicing(clean_img, is_monochrome)
        p_shot = self._expert_photon_shot_noise(clean_img)
        p_aberr = self._expert_chromatic_aberration(clean_img, is_monochrome)
        p_jpeg = self._expert_jpeg_block_grid(clean_img)
        p_entropy = self._expert_micro_texture_entropy(clean_img)

        experts = [
            ("1. Deep Hypersphere Anomaly (OOD Distance)", p_sphere, 1.35),
            ("2. Deep PatchCraft & FreqNet Energy", p_patch_freq, 1.35),
            ("3. Deep Learned Pattern (Calibrated)", p_linear, 1.10),
            ("4. Bokeh & Optical DOF Physics", p_bokeh, 1.05),
            ("5. Smooth-Surface Plasticity & Fake-Grain", p_plastic, 1.15),
            ("6. Color Gamut & Synthetic Hue Coupling", p_color, 0.70),
            ("7. Natural Scene Statistics (MSCN)", p_mscn, 1.05),
            ("8. Gradient Heavy-Tail & Edge Profile", p_grad, 1.00),
            ("9. 2D-FFT Radial Power Decay", p_fft, 0.95),
            ("10. Multi-Scale Upsampling Residual", p_upsample, 0.95),
            ("11. Bayer CFA Sensor Demosaicing", p_cfa, 1.10),
            ("12. Photon Shot-Noise Consistency", p_shot, 1.00),
            ("13. Optical Lens Chromatic Aberration", p_aberr, 0.90),
            ("14. JPEG 8x8 Grid Continuity", p_jpeg, 0.85),
            ("15. Micro-Texture Residual Entropy", p_entropy, 1.05),
        ]

        # ექსპერტი ითვლება "წითელ ფლაგად" (AI ანომალიად), თუ მისი ქულა >= 0.55
        flag_threshold = 0.55
        flagged_count = sum(1 for _, score, _ in experts if score >= flag_threshold)

        total_w = sum(w for _, _, w in experts)
        weighted_avg = sum(score * w for _, score, w in experts) / total_w

        # კვორუმის კალიბრაცია (Quorum Consensus Logic):
        # - თუ 15-დან მხოლოდ <= 4 ექსპერტი აფიქსირებს ანომალიას, სურათი რეალურია!
        # - თუ >= 7 ექსპერტი აფიქსირებს ანომალიას, სურათი მკაფიოდ AI-ია!
        if flagged_count <= 3:
            final_ai_prob = min(0.38, weighted_avg * 0.78)
        elif flagged_count == 4:
            final_ai_prob = min(0.45, weighted_avg * 0.88)
        elif flagged_count == 5:
            final_ai_prob = weighted_avg * 0.95
        elif flagged_count == 6:
            final_ai_prob = max(0.54, weighted_avg * 1.04)
        else:
            # 7 ან მეტი ფლაგი 15-დან -> მტკიცე AI კვორუმი
            quorum_boost = (flagged_count - 6) * 0.045
            final_ai_prob = min(0.985, max(0.62, weighted_avg + quorum_boost))

        return {
            "ai_prob": float(final_ai_prob),
            "real_prob": float(1.0 - final_ai_prob),
            "flagged_count": flagged_count,
            "total_experts": len(experts),
            "experts": experts,
            "xr_vis": xr_vis,
            "xp_vis": xp_vis,
            "orig_size": (orig_w, orig_h),
            "is_monochrome": is_monochrome,
            "canvas_ratio": canvas_ratio,
            "mean_dist": mean_dist,
            "mean_rad": mean_rad,
        }