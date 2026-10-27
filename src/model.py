import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NPRLayer(nn.Module):
    """
    CVPR 2024: Neighboring Pixel Relationships (NPR).
    გამომავალი არხები: 15 (სუფთა 1:1 [0, 1] RGB-ზე).
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
        return torch.cat([dir_diffs, combined_3ch.to(dir_diffs.dtype)], dim=1) * 4.0


class LPDLayer(nn.Module):
    """
    NeurIPS 2025 (FerretNet): Local Pixel Dependencies (LPD) via Zero-Masked Median MRF.
    გამომავალი არხები: 6.
    """
    def __init__(self) -> None:
        super().__init__()
        neighbor_indices = [0, 1, 2, 3, 5, 6, 7, 8]
        weights = torch.zeros((8, 1, 3, 3), dtype=torch.float32)
        for idx, pos in enumerate(neighbor_indices):
            r, c = divmod(pos, 3)
            weights[idx, 0, r, c] = 1.0
        self.register_buffer("unfold_weights", weights.repeat(3, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        neighbors = F.conv2d(x, self.unfold_weights.to(x.dtype), padding=1, groups=3).view(b, c, 8, h, w)
        zero_center = torch.zeros((b, c, 1, h, w), dtype=neighbors.dtype, device=x.device)
        extended = torch.cat([neighbors, zero_center], dim=2)
        median_rec = extended.median(dim=2).values

        lpd_signed = (x.to(median_rec.dtype) - median_rec) * 4.0
        lpd_abs = torch.abs(lpd_signed)
        return torch.cat([lpd_signed, lpd_abs], dim=1)


class HaarWaveletLayer(nn.Module):
    """
    KDD 2025 (SAFE): 2D Discrete Haar Wavelet Transform (DWT).
    გამომავალი არხები: 9.
    """
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
        return dwt_up * 4.0


class SRMFilterBank(nn.Module):
    """
    PatchCraft / Fridrich Steganalysis Rich Model (SRM) 15 მაღალი რიგის ფილტრი.
    გამომავალი არხები: 15.
    """
    def __init__(self) -> None:
        super().__init__()
        srm = torch.zeros((5, 1, 5, 5), dtype=torch.float32)
        srm[0, 0, 2, 2], srm[0, 0, 2, 3] = -1.0, 1.0
        srm[1, 0, 2, 1], srm[1, 0, 2, 2], srm[1, 0, 2, 3] = 1.0, -2.0, 1.0
        srm[2, 0, 1, 2], srm[2, 0, 2, 2], srm[2, 0, 3, 2] = 1.0, -2.0, 1.0
        square_3x3 = torch.tensor(
            [[-1.0, 2.0, -1.0], [2.0, -4.0, 2.0], [-1.0, 2.0, -1.0]], dtype=torch.float32
        ) / 4.0
        srm[3, 0, 1:4, 1:4] = square_3x3
        square_5x5 = torch.tensor(
            [
                [-1.0, 2.0, -2.0, 2.0, -1.0],
                [2.0, -6.0, 8.0, -6.0, 2.0],
                [-2.0, 8.0, -12.0, 8.0, -2.0],
                [2.0, -6.0, 8.0, -6.0, 2.0],
                [-1.0, 2.0, -2.0, 2.0, -1.0],
            ],
            dtype=torch.float32,
        ) / 12.0
        srm[4, 0] = square_5x5
        self.register_buffer("srm_kernels", srm.repeat(3, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.conv2d(x, self.srm_kernels.to(x.dtype), padding=2, groups=3)
        return torch.clamp(out * 4.0, min=-3.0, max=3.0)


class L1BayarConv2d(nn.Module):
    """
    L1-ნორმალიზებული Per-Channel Zero-Sum Constrained Convolution.
    გამომავალი არხები: 12.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 12, kernel_size: int = 5) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

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
        return F.conv2d(x, w_constrained, padding=self.kernel_size // 2) * 4.0


class DemosaicingChromaExtractor(nn.Module):
    """
    2026 (Color Matters / Bayer CFA Forensics):
    გამომავალი არხები: 7.
    """
    def __init__(self) -> None:
        super().__init__()
        lap = torch.tensor(
            [[[[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]]], dtype=torch.float32
        ) / 4.0
        checker = torch.tensor(
            [[[[0.25, -0.25, 0.25], [-0.25, 0.0, -0.25], [0.25, -0.25, 0.25]]]], dtype=torch.float32
        )
        self.register_buffer("lap_kernels", lap.repeat(3, 1, 1, 1))
        self.register_buffer("checker_kernels", checker.repeat(3, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        chroma_diffs = torch.cat([r - g, g - b, b - r], dim=1)

        chroma_lap = F.conv2d(chroma_diffs, self.lap_kernels.to(x.dtype), padding=1, groups=3) * 4.0
        cfa_periodic = F.conv2d(x, self.checker_kernels.to(x.dtype), padding=1, groups=3) * 4.0

        lum = 0.299 * r + 0.587 * g + 0.114 * b
        local_mean = F.avg_pool2d(lum, kernel_size=3, stride=1, padding=1)
        noise_sq = (lum - local_mean).pow(2)
        local_var = F.avg_pool2d(noise_sq, kernel_size=5, stride=1, padding=2)
        shot_ratio = (torch.sqrt(local_var + 1e-6) / (local_mean + 0.05)).to(chroma_lap.dtype)

        return torch.cat([chroma_lap, cfa_periodic, shot_ratio], dim=1)


class ForensicFrontEnd(nn.Module):
    """
    6-სვეტიანი ფორენზიკული შესასვლელი:
    - იღებს 64 ფორენზიკულ არხს სრულ 256x256 (1:1 პიქსელურ) რეზოლუციაზე.
    - 5 ფიქსირებული მათემატიკური ოპერატორი ეშვება torch.no_grad()-ში (ზოგავს ~10 GB VRAM-ს!).
    - ნაკერების დახშობის (seam_mask) შემდეგ გადადის 96-არხიან 128x128 სივრცეში.
    """
    def __init__(self, tile_size: int = 32, out_channels: int = 96) -> None:
        super().__init__()
        self.npr = NPRLayer()                        # 15 არხი
        self.lpd = LPDLayer()                        # 6 არხი
        self.haar = HaarWaveletLayer()               # 9 არხი
        self.srm = SRMFilterBank()                   # 15 არხი
        self.chroma = DemosaicingChromaExtractor()   # 7 არხი
        self.bayar = L1BayarConv2d(3, 12, 5)         # 12 არხი (სწავლებადი)

        seam_mask = torch.ones((1, 1, 256, 256), dtype=torch.float32)
        for idx in range(256):
            rem = idx % tile_size
            if rem < 2 or rem >= tile_size - 2:
                seam_mask[:, :, idx, :] = 0.0
                seam_mask[:, :, :, idx] = 0.0
        self.register_buffer("seam_mask", seam_mask)

        self.stem_proj = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = torch.bfloat16 if torch.is_autocast_enabled() else x.dtype
        x_in = x.to(target_dtype)

        # ფიქსირებული 52 არხი ითვლება გრადიენტების შენახვის გარეშე
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

        bayar_feats = self.bayar(x_in).to(target_dtype)
        feats = torch.cat([fixed_feats, bayar_feats], dim=1)

        if feats.shape[-2:] == self.seam_mask.shape[-2:]:
            feats = feats * self.seam_mask.to(target_dtype)
        else:
            mask_resized = F.interpolate(self.seam_mask, size=feats.shape[-2:], mode="nearest")
            feats = feats * mask_resized.to(target_dtype)

        out = self.stem_proj(feats)
        return out.to(target_dtype)


class SEAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(16, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * w


class MultiScaleForensicBlock(nn.Module):
    """
    მძლავრი ჯგუფური კონვოლუციური ბლოკი (groups=2):
    აერთიანებს ლოკალურ (dilation=1) და გაფართოებულ (dilation=2) სივრცით-არხულ კავშირებს.
    ინარჩუნებს BFloat16 ტიპს GroupNorm-ის შემდეგაც, რათა VRAM არ გადაივსოს.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        mid_ch = out_ch // 2

        self.conv_local = nn.Conv2d(
            in_ch, mid_ch, kernel_size=3, stride=stride, padding=1, groups=2, bias=False
        )
        self.conv_dilated = nn.Conv2d(
            in_ch, mid_ch, kernel_size=3, stride=stride, padding=2, dilation=2, groups=2, bias=False
        )
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.act1 = nn.SiLU(inplace=True)

        self.pw = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.se = SEAttention(out_ch)
        self.act2 = nn.SiLU(inplace=True)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride) if stride > 1 else nn.Identity(),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.GroupNorm(8, out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = x.dtype
        residual = self.shortcut(x).to(target_dtype)

        out = torch.cat([self.conv_local(x), self.conv_dilated(x)], dim=1)
        out = self.act1(self.norm1(out).to(target_dtype))

        out = self.pw(out)
        out = self.drop(self.norm2(out).to(target_dtype))
        out = self.se(out)
        return self.act2(out + residual)


class FreqNetSpectralBlock(nn.Module):
    """
    AAAI 2024 (FreqNet): სპექტრული კონვოლუცია ამპლიტუდისა და ფაზის სპექტრზე (64x64 რეზოლუციაზე).
    """
    def __init__(self, channels: int = 256, out_dim: int = 192) -> None:
        super().__init__()
        self.amp_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )
        self.phase_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(channels * 2, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, out_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        orig_dtype = feat.dtype
        with torch.amp.autocast("cuda", enabled=False):
            feat_f32 = feat.to(torch.float32)
            h, w = feat_f32.shape[-2:]

            fft_c = torch.fft.rfft2(feat_f32, norm="ortho")
            amp = torch.abs(fft_c) + 1e-6
            phase = torch.angle(fft_c)

            amp_mod = F.softplus(amp + self.amp_conv(amp))
            phase_mod = phase + 0.2 * torch.tanh(self.phase_conv(phase))

            real = amp_mod * torch.cos(phase_mod)
            imag = amp_mod * torch.sin(phase_mod)
            fft_mod = torch.complex(real, imag)

            spatial_rec = torch.fft.irfft2(fft_mod, s=(h, w), norm="ortho")
            log_amp_pooled = F.interpolate(
                torch.log1p(amp_mod), size=(h, w), mode="bilinear", align_corners=False
            )

            combined = torch.cat([spatial_rec, log_amp_pooled], dim=1)
            out = self.spatial_proj(combined)
            pooled = F.adaptive_avg_pool2d(out, 1).flatten(1)

        return pooled.to(orig_dtype)


class ArtifactNetV2(nn.Module):
    """
    ArtifactNet v2.5 Ultra (~7.2M Parameters — 100% From Scratch):
    მძლავრი ჯგუფური კონვოლუციური ფორენზიკული არქიტექტურა 3 ექსპერტული შტოთი
    და SimLBR / DeepSVDD ასიმეტრიული Real-Camera ჰიპერსფეროთი.
    """
    def __init__(self, dropout: float = 0.25) -> None:
        super().__init__()
        self.frontend = ForensicFrontEnd(tile_size=32, out_channels=96)  # (2B, 96, 128, 128)

        self.stage1 = nn.Sequential(
            MultiScaleForensicBlock(96, 128, stride=1, dropout=0.05),
            MultiScaleForensicBlock(128, 128, stride=1, dropout=0.05),
        )
        self.stage2 = nn.Sequential(
            MultiScaleForensicBlock(128, 256, stride=2, dropout=0.10),
            MultiScaleForensicBlock(256, 256, stride=1, dropout=0.10),
        )

        # FreqNet სპექტრული შტო მუშაობს 64x64 რეზოლუციაზე (256 არხით)
        self.spectral_branch = FreqNetSpectralBlock(channels=256, out_dim=192)

        # PatchCraft Rich-vs-Poor კონტრასტული შტო (256 -> 192)
        self.contrast_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.Hardtanh(min_val=-3.0, max_val=3.0),
            nn.Conv2d(256, 192, kernel_size=1, bias=False),
            nn.GroupNorm(8, 192),
            nn.SiLU(inplace=True),
        )

        self.stage3 = nn.Sequential(
            MultiScaleForensicBlock(256, 384, stride=2, dropout=0.15),
            MultiScaleForensicBlock(384, 384, stride=1, dropout=0.15),
        )
        self.stage4 = nn.Sequential(
            MultiScaleForensicBlock(384, 512, stride=2, dropout=0.20),
            MultiScaleForensicBlock(512, 512, stride=1, dropout=0.20),
        )

        # Spatial Rich (512 Avg + 512 Max = 1024) + Contrast (192) + Spectral (192) = 1408
        self.embedding_proj = nn.Sequential(
            nn.LayerNorm(1408),
            nn.Dropout(dropout),
            nn.Linear(1408, 256, bias=False),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 128, bias=False),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
        )

        self.linear_head = nn.Linear(128, 1)

        self.register_buffer("real_center", torch.zeros(128, dtype=torch.float32))
        self.register_buffer("center_initialized", torch.tensor(False, dtype=torch.bool))
        self.log_radius_sq = nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=True)
        self.sphere_scale = nn.Parameter(torch.tensor(2.0, dtype=torch.float32), requires_grad=True)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def _update_real_center(self, z: torch.Tensor, targets: torch.Tensor) -> None:
        real_mask = targets.view(-1) < 0.5
        if not torch.any(real_mask):
            return
        z_real_mean = z[real_mask].detach().to(torch.float32).mean(dim=0)
        if not bool(self.center_initialized.item()):
            self.real_center.copy_(z_real_mean)
            self.center_initialized.fill_(True)
        else:
            self.real_center.mul_(0.98).add_(z_real_mean, alpha=0.02)

    def _AutoSplitRichPoor(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        ps = 32
        if h < ps * 2 or w < ps * 2:
            return x, x

        nh, nw = h // ps, w // ps
        n_patches = nh * nw
        patches = x[:, :, : nh * ps, : nw * ps].view(b, c, nh, ps, nw, ps)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous().view(b, n_patches, c, ps, ps)

        dh = (patches[:, :, :, :, 1:] - patches[:, :, :, :, :-1]).abs().mean(dim=(2, 3, 4))
        dv = (patches[:, :, :, 1:, :] - patches[:, :, :, :-1, :]).abs().mean(dim=(2, 3, 4))
        dd1 = (patches[:, :, :, 1:, 1:] - patches[:, :, :, :-1, :-1]).abs().mean(dim=(2, 3, 4))
        dd2 = (patches[:, :, :, 1:, :-1] - patches[:, :, :, :-1, 1:]).abs().mean(dim=(2, 3, 4))
        l_div = dh + dv + dd1 + dd2

        sort_idx = torch.argsort(l_div, dim=1, descending=True)
        k = max(1, n_patches // 3)
        rich_idx = sort_idx[:, :k]
        poor_idx = sort_idx[:, -k:]

        repeats = math.ceil(n_patches / k)
        rich_full_idx = rich_idx.repeat(1, repeats)[:, :n_patches]
        poor_full_idx = poor_idx.repeat(1, repeats)[:, :n_patches]

        idx_expand_r = rich_full_idx.view(b, n_patches, 1, 1, 1).expand(-1, -1, c, ps, ps)
        idx_expand_p = poor_full_idx.view(b, n_patches, 1, 1, 1).expand(-1, -1, c, ps, ps)

        rich_patches = torch.gather(patches, 1, idx_expand_r)
        poor_patches = torch.gather(patches, 1, idx_expand_p)

        x_rich = (
            rich_patches.view(b, nh, nw, c, ps, ps)
            .permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view(b, c, nh * ps, nw * ps)
        )
        x_poor = (
            poor_patches.view(b, nh, nw, c, ps, ps)
            .permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view(b, c, nh * ps, nw * ps)
        )
        return x_rich, x_poor

    def forward_features(
        self,
        x_rich: torch.Tensor,
        x_poor: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x_rich.dtype == torch.uint8:
            x_rich = x_rich.to(torch.float32) / 255.0
        if x_poor is not None and x_poor.dtype == torch.uint8:
            x_poor = x_poor.to(torch.float32) / 255.0

        if x_poor is None:
            x_rich, x_poor = self._AutoSplitRichPoor(x_rich)

        b = x_rich.shape[0]
        x_both = torch.cat([x_rich, x_poor], dim=0)
        f0 = self.frontend(x_both)
        f1 = self.stage1(f0)
        f2 = self.stage2(f1)

        f2_rich, f2_poor = f2[:b], f2[b:]

        c_rich = self.contrast_head(f2_rich)
        c_poor = self.contrast_head(f2_poor)
        contrast_diff = c_rich - c_poor
        contrast_vec = F.adaptive_avg_pool2d(contrast_diff, 1).flatten(1)

        spectral_vec = self.spectral_branch(f2_rich)

        f3 = self.stage3(f2_rich)
        f4 = self.stage4(f3)
        avg_p = F.adaptive_avg_pool2d(f4, 1).flatten(1)
        max_p = F.adaptive_max_pool2d(f4, 1).flatten(1)
        spatial_vec = torch.cat([avg_p, max_p], dim=1)

        combined = torch.cat([spatial_vec, contrast_vec, spectral_vec], dim=1)
        z = self.embedding_proj(combined)

        if self.training and targets is not None:
            self._update_real_center(z, targets)

        center = self.real_center.to(z.dtype).unsqueeze(0)
        dist_sq = torch.mean((z - center).pow(2), dim=1, keepdim=True)
        radius_sq = F.softplus(self.log_radius_sq).to(z.dtype)
        sphere_logit = F.softplus(self.sphere_scale).to(z.dtype) * (dist_sq - radius_sq)

        linear_logit = self.linear_head(z)
        final_logit = linear_logit + 0.5 * sphere_logit
        return final_logit, dist_sq, radius_sq

    def forward(
        self,
        x_rich: torch.Tensor,
        x_poor: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        final_logit, dist_sq, radius_sq = self.forward_features(x_rich, x_poor, targets=targets)
        if return_aux:
            return final_logit, dist_sq, radius_sq
        return final_logit