import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NPRLayer(nn.Module):
    """
    Neighboring Pixel Relationships (NPR, 15 channels):
    - 12 directional first-order differences (H, V, Diag1, Diag2 across RGB)
    - 3 local 2x2 grid & avgpool-bilinear interpolation residuals
    Soft-bounded via tanh so high-contrast scene edges never drown out 1-2 LSB synthesis grids.
    """

    def __init__(self) -> None:
        super().__init__()
        kernels = torch.zeros((4, 1, 3, 3), dtype=torch.float32)
        kernels[0, 0, 1, 1], kernels[0, 0, 1, 2] = 1.0, -1.0
        kernels[1, 0, 1, 1], kernels[1, 0, 2, 1] = 1.0, -1.0
        kernels[2, 0, 1, 1], kernels[2, 0, 2, 2] = 1.0, -1.0
        kernels[3, 0, 1, 1], kernels[3, 0, 2, 0] = 1.0, -1.0
        self.register_buffer("dir_kernels", kernels.repeat(3, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dir_diffs = F.conv2d(x, self.dir_kernels.to(x.dtype), padding=1, groups=3)

        anchor = x[:, :, 0::2, 0::2]
        anchor_grid = F.interpolate(anchor, size=x.shape[-2:], mode="nearest")
        npr_grid_res = x - anchor_grid

        down = F.avg_pool2d(x, kernel_size=2, stride=2)
        up = F.interpolate(down, size=x.shape[-2:], mode="bilinear", align_corners=False)
        interp_res = x - up

        combined_3ch = 0.5 * (npr_grid_res + interp_res)
        raw = torch.cat([dir_diffs, combined_3ch.to(dir_diffs.dtype)], dim=1)
        return 2.5 * torch.tanh(raw * 2.0)


class LPDLayer(nn.Module):
    """
    Unbiased Local Pixel Dependencies (FerretNet-style, 6 channels):
    Computes exact 8-neighbor surround median (average of 4th and 5th sorted surround pixels)
    without zero-padding bias, returning signed and magnitude residuals.
    """

    def __init__(self) -> None:
        super().__init__()
        neighbor_indices = [0, 1, 2, 3, 5, 6, 7, 8]
        weights = torch.zeros((8, 1, 3, 3), dtype=torch.float32)
        for idx, pos in enumerate(neighbor_indices):
            r, col = divmod(pos, 3)
            weights[idx, 0, r, col] = 1.0
        self.register_buffer("unfold_weights", weights.repeat(3, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        neighbors = F.conv2d(x, self.unfold_weights.to(x.dtype), padding=1, groups=3).view(b, c, 8, h, w)
        sorted_n = neighbors.sort(dim=2).values
        median_rec = 0.5 * (sorted_n[:, :, 3] + sorted_n[:, :, 4])

        lpd_signed = 2.5 * torch.tanh((x.to(median_rec.dtype) - median_rec) * 2.5)
        lpd_abs = torch.abs(lpd_signed)
        return torch.cat([lpd_signed, lpd_abs], dim=1)


class HaarWaveletLayer(nn.Module):
    """2D Discrete Haar Wavelet High-Frequency Sub-bands (LH, HL, HH -> 9 channels)."""

    def __init__(self) -> None:
        super().__init__()
        lh = torch.tensor([[[[-0.5, -0.5], [0.5, 0.5]]]], dtype=torch.float32)
        hl = torch.tensor([[[[-0.5, 0.5], [-0.5, 0.5]]]], dtype=torch.float32)
        hh = torch.tensor([[[[0.5, -0.5], [-0.5, 0.5]]]], dtype=torch.float32)
        kernels = torch.cat([lh, hl, hh], dim=0).repeat(3, 1, 1, 1)
        self.register_buffer("haar_kernels", kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dwt = F.conv2d(x, self.haar_kernels.to(x.dtype), stride=2, padding=0, groups=3)
        dwt_up = F.interpolate(dwt, size=x.shape[-2:], mode="nearest")
        return 2.5 * torch.tanh(dwt_up * 2.0)


class SRMFilterBank(nn.Module):
    """Steganalysis Rich Model (SRM) 1st, 2nd, 3rd, and KV 5x5 high-order filters (15 channels)."""

    def __init__(self) -> None:
        super().__init__()
        srm = torch.zeros((5, 1, 5, 5), dtype=torch.float32)
        srm[0, 0, 2, 2], srm[0, 0, 2, 3] = -1.0, 1.0
        srm[1, 0, 2, 1], srm[1, 0, 2, 2], srm[1, 0, 2, 3] = 0.5, -1.0, 0.5
        srm[2, 0, 1, 2], srm[2, 0, 2, 2], srm[2, 0, 3, 2] = 0.5, -1.0, 0.5
        square_3x3 = (
            torch.tensor([[-1.0, 2.0, -1.0], [2.0, -4.0, 2.0], [-1.0, 2.0, -1.0]], dtype=torch.float32) / 4.0
        )
        srm[3, 0, 1:4, 1:4] = square_3x3
        square_5x5 = (
            torch.tensor(
                [
                    [-1.0, 2.0, -2.0, 2.0, -1.0],
                    [2.0, -6.0, 8.0, -6.0, 2.0],
                    [-2.0, 8.0, -12.0, 8.0, -2.0],
                    [2.0, -6.0, 8.0, -6.0, 2.0],
                    [-1.0, 2.0, -2.0, 2.0, -1.0],
                ],
                dtype=torch.float32,
            )
            / 12.0
        )
        srm[4, 0] = square_5x5
        self.register_buffer("srm_kernels", srm.repeat(3, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.conv2d(x, self.srm_kernels.to(x.dtype), padding=2, groups=3)
        return torch.clamp(out * 3.5, min=-3.0, max=3.0)


class L1BayarConv2d(nn.Module):
    """
    Per-Channel Zero-Sum L1-Normalized Constrained Forensic Convolution (12 channels).
    Strictly annihilates constant brightness and smooth color gradients while learning
    micro-texture synthesis & sensor residual filters from scratch.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 12, kernel_size: int = 5) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1)

        mask = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32)
        mask[:, :, kernel_size // 2, kernel_size // 2] = 0.0
        self.register_buffer("surround_mask", mask)
        self.register_buffer("center_mask", 1.0 - mask)

    def _get_constrained_weights(self) -> torch.Tensor:
        w_surround = self.weight * self.surround_mask
        l1_norm = w_surround.abs().sum(dim=(2, 3), keepdim=True) + 1e-6
        w_norm = w_surround / l1_norm
        signed_sum = w_norm.sum(dim=(2, 3), keepdim=True)
        return w_norm - self.center_mask * signed_sum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_constrained = self._get_constrained_weights().to(x.dtype)
        raw = F.conv2d(x, w_constrained, padding=self.kernel_size // 2)
        return 2.5 * torch.tanh(raw * 2.0)


class DemosaicingChromaExtractor(nn.Module):
    """Cross-channel chroma Laplacian, Bayer 2x2 periodic response, and Poisson shot-noise ratio (7 channels)."""

    def __init__(self) -> None:
        super().__init__()
        lap = torch.tensor([[[[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]]], dtype=torch.float32) / 4.0
        checker = torch.tensor(
            [[[[0.25, -0.25, 0.25], [-0.25, 0.0, -0.25], [0.25, -0.25, 0.25]]]], dtype=torch.float32
        )
        self.register_buffer("lap_kernels", lap.repeat(3, 1, 1, 1))
        self.register_buffer("checker_kernels", checker.repeat(3, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        chroma_diffs = torch.cat([r - g, g - b, b - r], dim=1)

        chroma_lap = 2.5 * torch.tanh(F.conv2d(chroma_diffs, self.lap_kernels.to(x.dtype), padding=1, groups=3) * 2.5)
        cfa_periodic = 2.5 * torch.tanh(F.conv2d(x, self.checker_kernels.to(x.dtype), padding=1, groups=3) * 2.5)

        lum = 0.299 * r + 0.587 * g + 0.114 * b
        local_mean = F.avg_pool2d(lum, kernel_size=3, stride=1, padding=1)
        noise_sq = (lum - local_mean).pow(2)
        local_var = F.avg_pool2d(noise_sq, kernel_size=5, stride=1, padding=2)
        # True Poisson shot-noise law scales as sqrt(luminance)
        shot_ratio = (torch.sqrt(local_var + 1e-6) / torch.sqrt(torch.clamp(local_mean, min=0.02))).to(chroma_lap.dtype)
        shot_ratio = torch.clamp(shot_ratio * 4.0, 0.0, 3.0)

        return torch.cat([chroma_lap, cfa_periodic, shot_ratio], dim=1)


class ForensicFrontEnd(nn.Module):
    """
    6-Branch Forensic Stem (64 residual channels at 256x256 -> 96 channels at 128x128).
    Includes interior 32x32 tile-seam masking so patch mosaics never introduce artificial boundary edges.
    """

    def __init__(self, tile_size: int = 32, img_size: int = 256, out_channels: int = 96) -> None:
        super().__init__()
        self.npr = NPRLayer()
        self.lpd = LPDLayer()
        self.haar = HaarWaveletLayer()
        self.srm = SRMFilterBank()
        self.chroma = DemosaicingChromaExtractor()
        self.bayar = L1BayarConv2d(3, 12, 5)

        # Mask 2px on each side of INTERNAL tile boundaries (leaving outer image perimeter intact)
        seam_mask = torch.ones((1, 1, img_size, img_size), dtype=torch.float32)
        for seam in range(tile_size, img_size, tile_size):
            lo = max(0, seam - 2)
            hi = min(img_size, seam + 2)
            seam_mask[:, :, lo:hi, :] = 0.0
            seam_mask[:, :, :, lo:hi] = 0.0
        self.register_buffer("seam_mask", seam_mask)

        self.stem_proj = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, apply_seam_mask: bool = True) -> torch.Tensor:
        target_dtype = torch.bfloat16 if (x.is_cuda and torch.is_autocast_enabled()) else x.dtype
        x_in = x.to(target_dtype)

        with torch.no_grad():
            fixed_feats = torch.cat(
                [
                    self.npr(x_in),
                    self.lpd(x_in),
                    self.haar(x_in),
                    self.srm(x_in),
                    self.chroma(x_in),
                ],
                dim=1,
            ).to(target_dtype)
            if apply_seam_mask:
                mask = (
                    self.seam_mask
                    if fixed_feats.shape[-2:] == self.seam_mask.shape[-2:]
                    else F.interpolate(self.seam_mask, size=fixed_feats.shape[-2:], mode="nearest")
                ).to(target_dtype)
                fixed_feats.mul_(mask)

        bayar_feats = self.bayar(x_in).to(target_dtype)
        if apply_seam_mask:
            bayar_feats = bayar_feats * mask

        feats = torch.cat([fixed_feats, bayar_feats], dim=1)
        return self.stem_proj(feats).to(target_dtype)


class SpectralRingDropout(nn.Module):
    """
    GPU-Accelerated Anti-Shortcut Spectral Regularizer:
    Randomly attenuates 1-2 narrow radial frequency bands (and optional angular wedges) in 2D FFT
    identically across matched (Real, AI) pairs during training. Prevents the model from latching onto
    a single generator's deconvolution spike and forces broadband multi-scale learning.
    """

    def __init__(self, img_size: int = 256, prob: float = 0.50) -> None:
        super().__init__()
        self.prob = prob
        fy = torch.fft.fftfreq(img_size, d=1.0) * 2.0  # [-1, 1]
        fx = torch.fft.rfftfreq(img_size, d=1.0) * 2.0  # [0, 1]
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        rad = torch.sqrt(xx * xx + yy * yy)  # [0, sqrt(2)]
        ang = torch.atan2(yy.abs(), xx + 1e-6)  # [0, pi/2]
        self.register_buffer("rad_grid", rad.view(1, 1, img_size, fx.numel()))
        self.register_buffer("ang_grid", ang.view(1, 1, img_size, fx.numel()))

    @torch.no_grad()
    def forward(
        self,
        x_global: torch.Tensor,
        x_rich: torch.Tensor,
        x_poor: torch.Tensor,
        paired_batch: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training or self.prob <= 0.0:
            return x_global, x_rich, x_poor

        b, _, h, w = x_global.shape
        if h != self.rad_grid.shape[2]:
            return x_global, x_rich, x_poor

        dev = x_global.device
        group_n = (b // 2) if (paired_batch and b % 2 == 0 and b >= 2) else b

        active = (torch.rand((group_n, 1, 1, 1), device=dev) < self.prob).to(torch.float32)
        if active.sum() == 0:
            return x_global, x_rich, x_poor

        # Ring 1: mid-to-high frequency band [0.16, 0.98] (never touches DC/low-freq scene luminance)
        c1 = torch.empty((group_n, 1, 1, 1), device=dev).uniform_(0.18, 0.92)
        w1 = torch.empty((group_n, 1, 1, 1), device=dev).uniform_(0.05, 0.16)
        atten1 = torch.empty((group_n, 1, 1, 1), device=dev).uniform_(0.05, 0.35)

        # Ring 2 (50% chance when active)
        use_r2 = (torch.rand((group_n, 1, 1, 1), device=dev) < 0.55).to(torch.float32)
        c2 = torch.empty((group_n, 1, 1, 1), device=dev).uniform_(0.22, 0.96)
        w2 = torch.empty((group_n, 1, 1, 1), device=dev).uniform_(0.04, 0.14)
        atten2 = torch.empty((group_n, 1, 1, 1), device=dev).uniform_(0.05, 0.40)

        rad = self.rad_grid
        ring1 = torch.exp(-0.5 * ((rad - c1) / w1).pow(2))
        ring2 = torch.exp(-0.5 * ((rad - c2) / w2).pow(2)) * use_r2

        mask = (1.0 - active * (1.0 - atten1) * ring1) * (1.0 - active * (1.0 - atten2) * ring2)

        # Optional cardinal-vs-diagonal angular notch (prevents separable-grid shortcut)
        use_ang = (torch.rand((group_n, 1, 1, 1), device=dev) < 0.30).to(torch.float32) * active
        ang_c = torch.empty((group_n, 1, 1, 1), device=dev).uniform_(0.0, math.pi * 0.5)
        ang_notch = torch.exp(-0.5 * ((self.ang_grid - ang_c) / 0.18).pow(2)) * (rad > 0.20).to(torch.float32)
        mask = mask * (1.0 - 0.70 * use_ang * ang_notch)

        if group_n != b:
            mask = mask.repeat_interleave(2, dim=0)

        def _apply(t: torch.Tensor) -> torch.Tensor:
            orig_dtype = t.dtype
            tf = t.to(torch.float32)
            spec = torch.fft.rfft2(tf, norm="ortho")
            out = torch.fft.irfft2(spec * mask, s=(h, w), norm="ortho")
            return torch.clamp(out, 0.0, 1.0).to(orig_dtype)

        return _apply(x_global), _apply(x_rich), _apply(x_poor)


class ScaleFreePhysicsExtractor(nn.Module):
    """
    Scale-Free & Internet-Compression-Resilient Physical Forensics Engine (Zero BatchNorm, Monochrome-Safe):
    Produces:
      1. physics_maps: (B, 8, 256, 256) bounded spatial residual maps on x_global
      2. physics_vec:  (B, 28) dimensionless ratios, correlations, log-slopes, kurtoses, and meso-scale
                       compression-survivable descriptors compressed smoothly with asinh().
    """

    FEATURE_NAMES = [
        "MSCN Luminance Std",
        "MSCN Excess Kurtosis",
        "MSCN Skewness",
        "Rich/Poor Luma HF Log-Ratio",
        "Rich/Poor Chroma HF Log-Ratio",
        "Rich-Poor Luma/Chroma Differential",
        "FFT Log-Power Slope (Low->Mid)",
        "FFT Log-Power Slope (Mid->High)",
        "FFT Log-Power Slope (High->Ultra)",
        "FFT Cardinal/Diagonal Anisotropy (Mid)",
        "FFT Cardinal/Diagonal Anisotropy (High)",
        "Poisson Shot-Noise Bright/Mid Ratio",
        "Poisson Shot-Noise Mid/Shadow Ratio",
        "Luminance-Noise Variance Coupling",
        "YCrCb Luma/Chroma HF Coupling",
        "CFA R-G vs G-B Residual Correlation",
        "Optical Luma-Chroma Edge Alignment",
        "Wavelet Scale 1->2 Energy Ratio",
        "Wavelet Scale 2->3 Energy Ratio",
        "Wavelet Cross-Scale Parent-Child Corr",
        "Gradient Heavy-Tail P95/Mean Ratio",
        "Laplacian Excess Kurtosis",
        "2x2 Grid Diagonal Anisotropy",
        "Smooth-Region Residual Floor Ratio",
        "8x8 Block Boundary/Interior Step Ratio",
        "Meso-2x Rich/Poor Texture Log-Ratio",
        "Meso-4x vs Fine-1x Edge Acutance Ratio",
        "Radial Lens Center/Corner Sharpness Ratio",
    ]

    def __init__(self, img_size: int = 256) -> None:
        super().__init__()
        coords = torch.arange(7, dtype=torch.float32) - 3.0
        g1d = torch.exp(-(coords ** 2) / (2.0 * 1.166 ** 2))
        g2d = g1d.outer(g1d)
        g2d = (g2d / g2d.sum()).view(1, 1, 7, 7)
        self.register_buffer("gauss7", g2d)

        sx = torch.tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]], dtype=torch.float32) / 4.0
        sy = torch.tensor([[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]], dtype=torch.float32) / 4.0
        lap = torch.tensor([[[[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]]], dtype=torch.float32) / 4.0
        self.register_buffer("sobel_x", sx)
        self.register_buffer("sobel_y", sy)
        self.register_buffer("laplacian", lap)

        w_ll = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]], dtype=torch.float32)
        w_hf = torch.tensor(
            [
                [[[-0.5, -0.5], [0.5, 0.5]]],
                [[[-0.5, 0.5], [-0.5, 0.5]]],
                [[[0.5, -0.5], [-0.5, 0.5]]],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("w_ll", w_ll)
        self.register_buffer("w_hf", w_hf)

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, img_size),
            torch.linspace(-1.0, 1.0, img_size),
            indexing="ij",
        )
        rad = torch.sqrt(xx * xx + yy * yy)
        ang = torch.atan2(yy.abs(), xx.abs() + 1e-6)  # [0, pi/2]
        self.register_buffer("radial_grid", rad.view(1, 1, img_size, img_size))
        # Cardinal axes: near 0 or pi/2; Diagonal axes: near pi/4
        is_cardinal = ((ang < 0.26) | (ang > (math.pi * 0.5 - 0.26))).to(torch.float32)
        is_diagonal = ((ang > (math.pi * 0.25 - 0.26)) & (ang < (math.pi * 0.25 + 0.26))).to(torch.float32)
        self.register_buffer("cardinal_mask", is_cardinal.view(1, 1, img_size, img_size))
        self.register_buffer("diagonal_mask", is_diagonal.view(1, 1, img_size, img_size))

        win1d = torch.hann_window(img_size, dtype=torch.float32)
        self.register_buffer("hann2d", win1d.outer(win1d).view(1, 1, img_size, img_size))

    @staticmethod
    def _batch_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        af = a.flatten(1)
        bf = b.flatten(1)
        af = af - af.mean(dim=1, keepdim=True)
        bf = bf - bf.mean(dim=1, keepdim=True)
        num = (af * bf).sum(dim=1, keepdim=True)
        den = torch.sqrt((af * af).sum(dim=1, keepdim=True) * (bf * bf).sum(dim=1, keepdim=True) + 1e-6)
        return torch.clamp(num / den, -1.0, 1.0)

    @torch.no_grad()
    def forward(
        self,
        x_global: torch.Tensor,
        x_rich: torch.Tensor,
        x_poor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device_type = "cuda" if x_global.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            xg = torch.clamp(x_global.to(torch.float32), 0.0, 1.0)
            xr = torch.clamp(x_rich.to(torch.float32), 0.0, 1.0)
            xp = torch.clamp(x_poor.to(torch.float32), 0.0, 1.0)
            bsz = xg.shape[0]

            r, g, bl = xg[:, 0:1], xg[:, 1:2], xg[:, 2:3]
            gray = 0.299 * r + 0.587 * g + 0.114 * bl
            cr = 0.500 * r - 0.419 * g - 0.081 * bl
            cb = -0.169 * r - 0.331 * g + 0.500 * bl

            # Smooth monochrome gate: 0.0 for grayscale/B&W photos, 1.0 for normal color photos
            chroma_spread = ((r - g).abs() + (g - bl).abs() + (r - bl).abs()).mean(dim=(2, 3), keepdim=True) / 3.0
            chroma_gate_map = torch.clamp(chroma_spread / 0.025, 0.0, 1.0)  # (B, 1, 1, 1)
            chroma_gate = chroma_gate_map.view(bsz, 1)

            # 1. MSCN Natural Scene Statistics
            mu = F.conv2d(gray, self.gauss7, padding=3)
            mu_sq = F.conv2d(gray * gray, self.gauss7, padding=3)
            sigma = torch.sqrt(torch.clamp(mu_sq - mu * mu, min=1e-6))
            mscn = (gray - mu) / (sigma + 0.012)

            # 2. Gradients & Laplacians
            gx = F.conv2d(gray, self.sobel_x, padding=1)
            gy = F.conv2d(gray, self.sobel_y, padding=1)
            grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
            lap_y = F.conv2d(gray, self.laplacian, padding=1)
            lap_cr = F.conv2d(cr, self.laplacian, padding=1)
            lap_cb = F.conv2d(cb, self.laplacian, padding=1)
            lap_rg = F.conv2d(g - r, self.laplacian, padding=1)
            lap_gb = F.conv2d(g - bl, self.laplacian, padding=1)

            # Edge-normalized spatial maps
            edge_denom = 1.0 + 5.0 * grad_mag
            luma_chroma_map = (
                torch.tanh((lap_y.abs() / (lap_cr.abs() + lap_cb.abs() + 0.008) - 2.2) * 0.5) * chroma_gate_map
            )
            cfa_rg_map = torch.clamp((lap_rg * 5.0) / edge_denom, -2.0, 2.0) * chroma_gate_map
            cfa_gb_map = torch.clamp((lap_gb * 5.0) / edge_denom, -2.0, 2.0) * chroma_gate_map

            smooth5 = F.avg_pool2d(gray, kernel_size=5, stride=1, padding=2)
            micro_noise = (gray - smooth5).abs()
            local_noise = F.avg_pool2d(micro_noise, kernel_size=7, stride=1, padding=3)
            global_noise_med = local_noise.flatten(1).mean(dim=1, keepdim=True).view(bsz, 1, 1, 1) + 1e-4
            # Relative local noise floor (scale-free ratio against image's own average noise!)
            rel_noise_floor = torch.tanh((local_noise / global_noise_med - 1.0) * 1.2) * torch.exp(-grad_mag * 8.0)

            shot_expected = torch.sqrt(torch.clamp(mu, min=0.02))
            shot_raw = local_noise / shot_expected
            shot_mean = shot_raw.flatten(1).mean(dim=1, keepdim=True).view(bsz, 1, 1, 1) + 1e-4
            shot_consistency_map = torch.tanh((shot_raw / shot_mean - 1.0) * 1.5)

            local_grad_mean = F.avg_pool2d(grad_mag, kernel_size=11, stride=1, padding=5) + 1e-4
            grad_heavytail_map = torch.tanh((grad_mag / local_grad_mean - 1.8) * 0.7)

            # Wavelet fine-vs-coarse coherence map
            hf1 = F.conv2d(gray, self.w_hf, stride=2).abs().sum(dim=1, keepdim=True)
            ll1 = F.conv2d(gray, self.w_ll, stride=2)
            hf2 = F.conv2d(ll1, self.w_hf, stride=2).abs().sum(dim=1, keepdim=True)
            ll2 = F.conv2d(ll1, self.w_ll, stride=2)
            hf3 = F.conv2d(ll2, self.w_hf, stride=2).abs().sum(dim=1, keepdim=True)

            hf1_d = F.avg_pool2d(hf1, kernel_size=2, stride=2)
            hf2_up = F.interpolate(
                torch.tanh((hf1_d / (hf2 + 0.005) - 1.0) * 0.8),
                size=xg.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            physics_maps = torch.cat(
                [
                    torch.clamp(mscn / 3.0, -2.0, 2.0),
                    luma_chroma_map,
                    cfa_rg_map,
                    cfa_gb_map,
                    rel_noise_floor,
                    shot_consistency_map,
                    grad_heavytail_map,
                    hf2_up,
                ],
                dim=1,
            )

            # --- 24 Dimensionless Scalar Ratios ---
            flat_mscn = mscn.flatten(1)
            m_mean = flat_mscn.mean(dim=1, keepdim=True)
            m_var = flat_mscn.var(dim=1, keepdim=True) + 1e-5
            m_std = torch.sqrt(m_var)
            m_kurt = ((flat_mscn - m_mean).pow(4).mean(dim=1, keepdim=True) / (m_var * m_var)) - 3.0
            m_skew = (flat_mscn - m_mean).pow(3).mean(dim=1, keepdim=True) / m_var.pow(1.5)

            # Intra-image Rich vs Poor HF ratios (cancels file codec & global ISO)
            yr = 0.299 * xr[:, 0:1] + 0.587 * xr[:, 1:2] + 0.114 * xr[:, 2:3]
            yp = 0.299 * xp[:, 0:1] + 0.587 * xp[:, 1:2] + 0.114 * xp[:, 2:3]
            crr = 0.500 * xr[:, 0:1] - 0.419 * xr[:, 1:2] - 0.081 * xr[:, 2:3]
            crp = 0.500 * xp[:, 0:1] - 0.419 * xp[:, 1:2] - 0.081 * xp[:, 2:3]

            hf_yr = F.conv2d(yr, self.laplacian, padding=1).flatten(1).std(dim=1, keepdim=True) + 1e-4
            hf_yp = F.conv2d(yp, self.laplacian, padding=1).flatten(1).std(dim=1, keepdim=True) + 1e-4
            hf_cr = F.conv2d(crr, self.laplacian, padding=1).flatten(1).std(dim=1, keepdim=True) + 1e-4
            hf_cp = F.conv2d(crp, self.laplacian, padding=1).flatten(1).std(dim=1, keepdim=True) + 1e-4

            rp_luma_log = torch.log(hf_yr / hf_yp)
            rp_chroma_log = torch.log(hf_cr / hf_cp) * chroma_gate
            rp_lc_diff = (torch.log(hf_yr / hf_cr) - torch.log(hf_yp / hf_cp)) * chroma_gate

            # 2D-FFT Radial Slopes & Cardinal-Diagonal Anisotropy
            gray_z = (gray - gray.mean(dim=(2, 3), keepdim=True)) * self.hann2d
            fft_mag = torch.abs(torch.fft.fftshift(torch.fft.fft2(gray_z), dim=(-2, -1))) + 1e-6
            log_fft = torch.log(fft_mag)

            rad = self.radial_grid
            b1 = ((rad >= 0.08) & (rad < 0.22)).to(torch.float32)
            b2 = ((rad >= 0.22) & (rad < 0.42)).to(torch.float32)
            b3 = ((rad >= 0.42) & (rad < 0.65)).to(torch.float32)
            b4 = ((rad >= 0.65) & (rad < 0.90)).to(torch.float32)

            e1 = (log_fft * b1).sum(dim=(2, 3)) / (b1.sum() + 1.0)
            e2 = (log_fft * b2).sum(dim=(2, 3)) / (b2.sum() + 1.0)
            e3 = (log_fft * b3).sum(dim=(2, 3)) / (b3.sum() + 1.0)
            e4 = (log_fft * b4).sum(dim=(2, 3)) / (b4.sum() + 1.0)

            slope_12 = e2 - e1
            slope_23 = e3 - e2
            slope_34 = e4 - e3

            c_mid = b2 * self.cardinal_mask
            d_mid = b2 * self.diagonal_mask
            c_hi = b3 * self.cardinal_mask
            d_hi = b3 * self.diagonal_mask
            aniso_mid = (log_fft * c_mid).sum(dim=(2, 3)) / (c_mid.sum() + 1.0) - (
                (log_fft * d_mid).sum(dim=(2, 3)) / (d_mid.sum() + 1.0)
            )
            aniso_hi = (log_fft * c_hi).sum(dim=(2, 3)) / (c_hi.sum() + 1.0) - (
                (log_fft * d_hi).sum(dim=(2, 3)) / (d_hi.sum() + 1.0)
            )

            # Poisson Shot-Noise Across Luminance Terciles
            shad_m = ((mu >= 0.05) & (mu < 0.28)).to(torch.float32)
            mid_m = ((mu >= 0.28) & (mu < 0.62)).to(torch.float32)
            bri_m = ((mu >= 0.62) & (mu <= 0.93)).to(torch.float32)

            n_shad = (micro_noise * shad_m).flatten(1).sum(dim=1, keepdim=True) / (shad_m.flatten(1).sum(dim=1, keepdim=True) + 1.0) + 1e-4
            n_mid = (micro_noise * mid_m).flatten(1).sum(dim=1, keepdim=True) / (mid_m.flatten(1).sum(dim=1, keepdim=True) + 1.0) + 1e-4
            n_bri = (micro_noise * bri_m).flatten(1).sum(dim=1, keepdim=True) / (bri_m.flatten(1).sum(dim=1, keepdim=True) + 1.0) + 1e-4

            shot_bm = torch.log(n_bri / n_mid) * (bri_m.flatten(1).mean(dim=1, keepdim=True) > 0.02).to(torch.float32)
            shot_ms = torch.log(n_mid / n_shad) * (shad_m.flatten(1).mean(dim=1, keepdim=True) > 0.02).to(torch.float32)
            lum_noise_corr = self._batch_corr(mu, local_noise)

            # Chroma-Luma Coupling & Optical Alignment
            std_ly = lap_y.flatten(1).std(dim=1, keepdim=True) + 1e-4
            std_lc = 0.5 * (lap_cr.flatten(1).std(dim=1, keepdim=True) + lap_cb.flatten(1).std(dim=1, keepdim=True)) + 1e-4
            yc_hf_ratio = (torch.log(std_ly / std_lc) - 1.0) * chroma_gate
            cfa_corr = self._batch_corr(lap_rg, lap_gb) * chroma_gate

            gx_c = F.conv2d(cr, self.sobel_x, padding=1).abs() + F.conv2d(cb, self.sobel_x, padding=1).abs()
            gy_c = F.conv2d(cr, self.sobel_y, padding=1).abs() + F.conv2d(cb, self.sobel_y, padding=1).abs()
            grad_c = gx_c + gy_c
            edge_align_corr = self._batch_corr(grad_mag, grad_c) * chroma_gate

            # Wavelet Multi-Scale Parent-Child Ratios & Correlation
            w_e1 = hf1.flatten(1).mean(dim=1, keepdim=True) + 1e-4
            w_e2 = hf2.flatten(1).mean(dim=1, keepdim=True) + 1e-4
            w_e3 = hf3.flatten(1).mean(dim=1, keepdim=True) + 1e-4
            wav_r12 = torch.log(w_e1 / w_e2)
            wav_r23 = torch.log(w_e2 / w_e3)
            wav_pc_corr = self._batch_corr(hf1_d, hf2)

            # Gradient Heavy-Tail & Laplacian Kurtosis
            grad_peaks = F.max_pool2d(grad_mag, kernel_size=8, stride=8).flatten(1).mean(dim=1, keepdim=True)
            grad_mean = grad_mag.flatten(1).mean(dim=1, keepdim=True) + 1e-4
            grad_tail = torch.log(grad_peaks / grad_mean + 1e-4)

            flat_lap = lap_y.flatten(1)
            lap_var = flat_lap.var(dim=1, keepdim=True) + 1e-6
            lap_kurt = (flat_lap.pow(4).mean(dim=1, keepdim=True) / (lap_var * lap_var) - 3.0) / 5.0

            # 2x2 Grid Diagonal Anisotropy & Smooth-Region Floor Ratio
            p00 = gray[:, :, 0::2, 0::2]
            p01 = gray[:, :, 0::2, 1::2]
            p10 = gray[:, :, 1::2, 0::2]
            p11 = gray[:, :, 1::2, 1::2]
            card_step = 0.5 * ((p00 - p01).abs().mean(dim=(2, 3)) + (p00 - p10).abs().mean(dim=(2, 3))) + 1e-4
            diag_step = (p00 - p11).abs().mean(dim=(2, 3))
            grid_aniso = (diag_step / card_step) - 1.4142

            smooth_mask = ((grad_mag < 0.06) & (gray > 0.08) & (gray < 0.92)).to(torch.float32)
            smooth_n = (micro_noise * smooth_mask).flatten(1).sum(dim=1, keepdim=True) / (
                smooth_mask.flatten(1).sum(dim=1, keepdim=True) + 1.0
            )
            all_n = micro_noise.flatten(1).mean(dim=1, keepdim=True) + 1e-4
            smooth_floor_ratio = torch.log((smooth_n + 1e-4) / all_n)

            # --- Compression-Survivable Meso-Scale & Optical Lens Descriptors (25..28) ---
            # 25. 8x8 DCT Block Boundary vs Interior step ratio (explicit codec/recompression gauge)
            dh_all = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs()
            dv_all = (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs()
            bnd_h = dh_all[:, :, :, 7::8].mean(dim=(2, 3))
            int_h = dh_all[:, :, :, 3::8].mean(dim=(2, 3)) + 1e-4
            bnd_v = dv_all[:, :, 7::8, :].mean(dim=(2, 3))
            int_v = dv_all[:, :, 3::8, :].mean(dim=(2, 3)) + 1e-4
            block8_ratio = 0.5 * (torch.log((bnd_h + 1e-4) / int_h) + torch.log((bnd_v + 1e-4) / int_v))

            # 26. Meso-2x Downsampled Rich/Poor Luma Log-Ratio (survives WhatsApp/WebP/2-hop resize)
            yr_2x = F.avg_pool2d(yr, kernel_size=2, stride=2)
            yp_2x = F.avg_pool2d(yp, kernel_size=2, stride=2)
            meso_yr = F.conv2d(yr_2x, self.laplacian, padding=1).flatten(1).std(dim=1, keepdim=True) + 1e-4
            meso_yp = F.conv2d(yp_2x, self.laplacian, padding=1).flatten(1).std(dim=1, keepdim=True) + 1e-4
            meso_rp_log = torch.log(meso_yr / meso_yp)

            # 27. Meso-4x vs Fine-1x Edge Acutance Consistency
            gray_4x = F.avg_pool2d(gray, kernel_size=4, stride=4)
            gx4 = F.conv2d(gray_4x, self.sobel_x, padding=1)
            gy4 = F.conv2d(gray_4x, self.sobel_y, padding=1)
            gmag4 = torch.sqrt(gx4 * gx4 + gy4 * gy4 + 1e-6)
            meso_acutance = torch.log((gmag4.flatten(1).mean(dim=1, keepdim=True) + 1e-4) / grad_mean)

            # 28. Radial Optical Center-vs-Corner Sharpness Ratio (real lens PSF falloff vs stationary AI kernel)
            center_m = (rad < 0.50).to(torch.float32)
            corner_m = (rad > 0.88).to(torch.float32)
            c_sharp = (grad_mag * center_m).flatten(1).sum(dim=1, keepdim=True) / (center_m.sum() + 1.0) + 1e-4
            k_sharp = (grad_mag * corner_m).flatten(1).sum(dim=1, keepdim=True) / (corner_m.sum() + 1.0) + 1e-4
            radial_lens_ratio = torch.log(c_sharp / k_sharp)

            raw_vec = torch.cat(
                [
                    m_std - 0.7,
                    m_kurt / 3.0,
                    m_skew,
                    rp_luma_log,
                    rp_chroma_log,
                    rp_lc_diff,
                    slope_12,
                    slope_23,
                    slope_34,
                    aniso_mid * 3.0,
                    aniso_hi * 3.0,
                    shot_bm,
                    shot_ms,
                    lum_noise_corr,
                    yc_hf_ratio,
                    cfa_corr,
                    edge_align_corr,
                    wav_r12,
                    wav_r23,
                    wav_pc_corr,
                    grad_tail,
                    lap_kurt,
                    grid_aniso * 3.0,
                    smooth_floor_ratio,
                    block8_ratio * 2.0,
                    meso_rp_log,
                    meso_acutance,
                    radial_lens_ratio,
                ],
                dim=1,
            )

            # Smooth unbounded compression via asinh: preserves sign & small values linearly, compresses extremes logarithmically
            physics_vec = torch.asinh(torch.nan_to_num(raw_vec, nan=0.0, posinf=10.0, neginf=-10.0))
            physics_maps = torch.nan_to_num(physics_maps, nan=0.0, posinf=2.0, neginf=-2.0)

        target_dtype = torch.bfloat16 if (x_global.is_cuda and torch.is_autocast_enabled()) else x_global.dtype
        return physics_maps.to(target_dtype), physics_vec.to(target_dtype)
