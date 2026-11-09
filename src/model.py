import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.forensics import (
    ForensicFrontEnd,
    ScaleFreePhysicsExtractor,
    SpectralRingDropout,
)

FAMILY_NAMES = [
    "Real Camera Photograph",
    "FLUX Family (FLUX.1 / 1.1-Pro / FLUX.2)",
    "OpenAI (GPT-Image & DALL-E 3)",
    "Midjourney (v6 / v7)",
    "Google (Nano-Banana & Imagen 3/4)",
    "Stable Diffusion 3 / 3.5",
    "SDXL & Photorealistic Fine-Tunes",
    "Stable Diffusion 1.x / 2.x",
    "Commercial Frontier (Seedream / Ideogram / Grok / Recraft)",
    "Next-Gen Open DiT (Qwen / HiDream / Kolors / Z-Image / Krea)",
    "AI Image-Edit / Video Synthesis",
    "Other Synthetic Generator",
]


def map_generator_to_family(label: int, model_name: str, item_type: str = "") -> int:
    """Maps (binary_label, generator_name, type) into one of 12 canonical forensic families."""
    if int(label) == 0:
        return 0
    m = (model_name or "").strip().lower()
    t = (item_type or "").strip().lower()

    if "edit" in m or "instructpix2pix" in m or "edit" in t or "video" in t or "wan-" in m or "veo" in m or "kling" in m or "hunyuan-video" in m or "ltxv" in m:
        return 10
    if "flux" in m:
        return 1
    if "gpt-image" in m or "chatgpt" in m or "dalle" in m or "dall-e" in m or "openai" in m:
        return 2
    if "midjourney" in m or m in ("mj", "mj6", "mj7"):
        return 3
    if "nano-banana" in m or "nanobanana" in m or "imagen" in m or "gemini" in m:
        return 4
    if "sd-3" in m or "sd3" in m or "stable-diffusion-3" in m:
        return 5
    if (
        "sdxl" in m
        or "sd-xl" in m
        or "xl-" in m
        or "pony" in m
        or "juggernaut" in m
        or "realvis" in m
        or "animagine" in m
        or "illustrious" in m
        or "novaanimexl" in m
        or "playground" in m
    ):
        return 6
    if (
        "sd-1" in m
        or "sd-2" in m
        or "sd1" in m
        or "sd2" in m
        or "v1-4" in m
        or "v1-5" in m
        or "dreamshaper" in m
        or "epicdream" in m
        or "realistic_vision" in m
        or "realistic-vision" in m
        or "sd-turbo" in m
    ):
        return 7
    if (
        "seedream" in m
        or "ideogram" in m
        or "grok" in m
        or "recraft" in m
        or "mystic" in m
        or "firefly" in m
        or "aurora" in m
        or "minimax" in m
    ):
        return 8
    if (
        "qwen" in m
        or "hidream" in m
        or "kolors" in m
        or "z-image" in m
        or "z image" in m
        or "krea" in m
        or "chroma" in m
        or "lumina" in m
        or "anima" in m
    ):
        return 9
    return 11


class DropPath(nn.Module):
    """Per-sample Stochastic Depth for residual blocks."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask


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
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        dropout: float = 0.08,
        drop_path: float = 0.05,
    ) -> None:
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
        self.drop_path = DropPath(drop_path)
        self.act2 = nn.SiLU(inplace=True)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True) if stride > 1 else nn.Identity(),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.GroupNorm(8, out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = torch.cat([self.conv_local(x), self.conv_dilated(x)], dim=1)
        out = self.act1(self.norm1(out))

        out = self.pw(out)
        out = self.drop(self.norm2(out))
        out = self.se(out)
        return self.act2(residual + self.drop_path(out))


class FreqNetSpectralBlock(nn.Module):
    """2D-FFT Amplitude & Phase Residual Modulation Block on Feature Maps."""

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
        dev_type = "cuda" if feat.is_cuda else "cpu"
        with torch.amp.autocast(device_type=dev_type, enabled=False):
            feat_f32 = feat.to(torch.float32)
            h, w = feat_f32.shape[-2:]

            fft_c = torch.fft.rfft2(feat_f32, norm="ortho")
            amp = torch.abs(fft_c) + 1e-6
            phase = torch.angle(fft_c)

            amp_mod = F.softplus(amp + self.amp_conv(amp))
            phase_mod = phase + 0.25 * torch.tanh(self.phase_conv(phase))

            fft_mod = torch.complex(amp_mod * torch.cos(phase_mod), amp_mod * torch.sin(phase_mod))
            spatial_rec = torch.fft.irfft2(fft_mod, s=(h, w), norm="ortho")
            log_amp_up = F.interpolate(
                torch.log1p(amp_mod), size=(h, w), mode="bilinear", align_corners=False
            )

            combined = torch.cat([spatial_rec, log_amp_up], dim=1)
            out = self.spatial_proj(combined)
            pooled = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return pooled.to(orig_dtype)


class ConvNeXtForensicBlock(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.08) -> None:
        super().__init__()
        self.dwconv7 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.dwconv_dil = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim, bias=False)
        self.norm = nn.GroupNorm(8, dim)
        self.pwconv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.act = nn.SiLU(inplace=True)
        self.pwconv2 = nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.dwconv7(x) + self.dwconv_dil(x)
        x = self.norm(x)
        x = self.act(self.pwconv1(x))
        x = self.pwconv2(x)
        return res + self.drop_path(x)


class MicroResidualStream(nn.Module):
    """
    Stream 1: Differential Rich-vs-Poor Micro & Meso Forensic Backbone.
    Compares high-texture patches against smooth midtone patches FROM THE SAME IMAGE:
    - Cancels global ISO level and social-platform codec (PNG vs WhatsApp JPEG vs WebP).
    - Exposes synthetic decoders' smooth-latent vs sharpened-edge discrepancy even after recompression.
    """

    def __init__(self, embed_dim: int = 224, dropout: float = 0.25) -> None:
        super().__init__()
        self.frontend = ForensicFrontEnd(tile_size=32, img_size=256, out_channels=96)
        self.stage1 = nn.Sequential(
            MultiScaleForensicBlock(96, 144, stride=2, dropout=0.05, drop_path=0.04),
            MultiScaleForensicBlock(144, 144, stride=1, dropout=0.05, drop_path=0.04),
        )
        self.stage2 = nn.Sequential(
            MultiScaleForensicBlock(144, 256, stride=2, dropout=0.08, drop_path=0.06),
            MultiScaleForensicBlock(256, 256, stride=1, dropout=0.08, drop_path=0.06),
        )
        self.contrast_proj = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.Hardtanh(min_val=-3.0, max_val=3.0),
            nn.Conv2d(256, 192, kernel_size=1, bias=False),
            nn.GroupNorm(8, 192),
            nn.SiLU(inplace=True),
        )
        self.spectral_block = FreqNetSpectralBlock(channels=256, out_dim=192)

        self.stage3 = nn.Sequential(
            MultiScaleForensicBlock(256, 384, stride=2, dropout=0.12, drop_path=0.08),
            MultiScaleForensicBlock(384, 384, stride=1, dropout=0.12, drop_path=0.10),
        )
        self.stage4 = nn.Sequential(
            MultiScaleForensicBlock(384, 512, stride=2, dropout=0.15, drop_path=0.12),
            MultiScaleForensicBlock(512, 512, stride=1, dropout=0.15, drop_path=0.12),
        )

        # 512*2 (Avg+Max) + 192 (diff_mean) + 256 (log_var_ratio) + 192 (spectral) = 1664
        self.proj = nn.Sequential(
            nn.LayerNorm(1664),
            nn.Dropout(dropout),
            nn.Linear(1664, 384, bias=False),
            nn.LayerNorm(384),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(384, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Linear(embed_dim, 1)

    def forward(
        self, x_rich: torch.Tensor, x_poor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = x_rich.shape[0]
        x_both = torch.cat([x_rich, x_poor], dim=0)
        f0 = self.frontend(x_both, apply_seam_mask=True)
        f1 = self.stage1(f0)
        f2 = self.stage2(f1)

        f2_rich, f2_poor = f2[:b], f2[b:]
        dtype = f2_rich.dtype

        c_rich = self.contrast_proj(f2_rich)
        c_poor = self.contrast_proj(f2_poor)
        diff_mean = F.adaptive_avg_pool2d(c_rich - c_poor, 1).flatten(1)

        var_r = f2_rich.to(torch.float32).var(dim=(2, 3)) + 1e-4
        var_p = f2_poor.to(torch.float32).var(dim=(2, 3)) + 1e-4
        log_var_ratio = torch.asinh(torch.log(var_r / var_p)).to(dtype)

        spec_vec = self.spectral_block(f2_rich)

        f3 = self.stage3(f2_rich)
        f4 = self.stage4(f3)
        avg_p = F.adaptive_avg_pool2d(f4, 1).flatten(1)
        max_p = F.adaptive_max_pool2d(f4, 1).flatten(1)

        combined = torch.cat([avg_p, max_p, diff_mean, log_var_ratio.to(avg_p.dtype), spec_vec.to(avg_p.dtype)], dim=1)
        emb = self.proj(combined)
        return self.head(emb), emb, f4


class WaveletPhaseSpectralStream(nn.Module):
    """
    Stream 2: Multi-Scale Wavelet Parent-Child Coherence + Per-Image Normalized 2D-FFT Phase/Amp Stream.
    Measures cross-scale phase alignment on full-scene `x_global` (survives resize & social compression).
    """

    def __init__(self, embed_dim: int = 224, dropout: float = 0.25) -> None:
        super().__init__()
        ll = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]], dtype=torch.float32)
        lh = torch.tensor([[[[-0.5, -0.5], [0.5, 0.5]]]], dtype=torch.float32)
        hl = torch.tensor([[[[-0.5, 0.5], [-0.5, 0.5]]]], dtype=torch.float32)
        hh = torch.tensor([[[[0.5, -0.5], [-0.5, 0.5]]]], dtype=torch.float32)
        self.register_buffer("w_ll", ll.repeat(3, 1, 1, 1))
        self.register_buffer("w_hf", torch.cat([lh, hl, hh], dim=0).repeat(3, 1, 1, 1))

        self.wav_enc1 = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
        )
        # +9 (hf2) +9 (parent-child ratio) = 64 + 18 = 82
        self.wav_enc2 = nn.Sequential(
            MultiScaleForensicBlock(64 + 18, 144, stride=1, dropout=0.06, drop_path=0.05),
        )
        # +9 (hf3) +9 (parent-child ratio) = 144 + 18 = 162
        self.wav_enc3 = nn.Sequential(
            MultiScaleForensicBlock(144 + 18, 256, stride=2, dropout=0.10, drop_path=0.08),
            MultiScaleForensicBlock(256, 256, stride=1, dropout=0.10, drop_path=0.08),
        )

        self.fft_net = nn.Sequential(
            nn.Conv2d(18, 80, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 80),
            nn.SiLU(inplace=True),
            MultiScaleForensicBlock(80, 160, stride=2, dropout=0.08, drop_path=0.05),
            MultiScaleForensicBlock(160, 256, stride=2, dropout=0.12, drop_path=0.08),
        )

        # Wavelet (256*2=512) + FFT (256*2=512) = 1024
        self.proj = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Dropout(dropout),
            nn.Linear(1024, 320, bias=False),
            nn.LayerNorm(320),
            nn.SiLU(inplace=True),
            nn.Linear(320, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Linear(embed_dim, 1)

    def _build_normalized_spectrum(self, x_global: torch.Tensor, x_rich: torch.Tensor) -> torch.Tensor:
        target_dtype = torch.bfloat16 if (x_global.is_cuda and torch.is_autocast_enabled()) else x_global.dtype
        dev_type = "cuda" if x_global.is_cuda else "cpu"
        with torch.amp.autocast(device_type=dev_type, enabled=False):
            both = torch.cat([x_global.to(torch.float32), x_rich.to(torch.float32)], dim=1)
            hp = both - F.avg_pool2d(both, kernel_size=3, stride=1, padding=1)

            fft_c = torch.fft.rfft2(hp, norm="ortho")[:, :, :, :-1]
            log_amp = torch.log1p(torch.abs(fft_c))
            phase = torch.angle(fft_c)

            pdx = F.pad(torch.sin(phase[:, :, :, 1:] - phase[:, :, :, :-1]), (0, 1, 0, 0))
            pdy = F.pad(torch.sin(phase[:, :, 1:, :] - phase[:, :, :-1, :]), (0, 0, 0, 1))

            spec = torch.cat([log_amp, pdx, pdy], dim=1)
            mean = spec.mean(dim=(2, 3), keepdim=True)
            std = spec.std(dim=(2, 3), keepdim=True) + 1e-5
            spec = torch.clamp(torch.nan_to_num((spec - mean) / std, nan=0.0), -4.0, 4.0)
        return spec.to(target_dtype)

    def forward(self, x_global: torch.Tensor, x_rich: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = torch.bfloat16 if (x_global.is_cuda and torch.is_autocast_enabled()) else x_global.dtype
        xg = x_global.to(dtype)

        with torch.no_grad():
            hf1 = 2.5 * torch.tanh(F.conv2d(xg, self.w_hf.to(dtype), stride=2, groups=3) * 2.0)
            ll1 = F.conv2d(xg, self.w_ll.to(dtype), stride=2, groups=3)
            hf2 = 2.5 * torch.tanh(F.conv2d(ll1, self.w_hf.to(dtype), stride=2, groups=3) * 2.0)
            ll2 = F.conv2d(ll1, self.w_ll.to(dtype), stride=2, groups=3)
            hf3 = 2.5 * torch.tanh(F.conv2d(ll2, self.w_hf.to(dtype), stride=2, groups=3) * 2.0)

            p12 = torch.tanh((F.avg_pool2d(hf1.abs(), 2, 2) / (hf2.abs() + 0.02) - 1.0) * 0.8).to(dtype)
            p23 = torch.tanh((F.avg_pool2d(hf2.abs(), 2, 2) / (hf3.abs() + 0.02) - 1.0) * 0.8).to(dtype)

        w1_d = self.wav_enc1(hf1)
        w2 = self.wav_enc2(torch.cat([w1_d, hf2.to(w1_d.dtype), p12.to(w1_d.dtype)], dim=1))
        w2_d = F.avg_pool2d(w2, 2, 2)
        w3 = self.wav_enc3(torch.cat([w2_d, hf3.to(w2_d.dtype), p23.to(w2_d.dtype)], dim=1))

        wav_pool = torch.cat(
            [F.adaptive_avg_pool2d(w3, 1).flatten(1), F.adaptive_max_pool2d(w3, 1).flatten(1)], dim=1
        )

        spec = self._build_normalized_spectrum(x_global, x_rich)
        f_fft = self.fft_net(spec)
        fft_pool = torch.cat(
            [F.adaptive_avg_pool2d(f_fft, 1).flatten(1), F.adaptive_max_pool2d(f_fft, 1).flatten(1)], dim=1
        )

        emb = self.proj(torch.cat([wav_pool, fft_pool], dim=1))
        return self.head(emb), emb


class GlobalConsistencyStream(nn.Module):
    """
    Stream 3: Compression-Resilient Meso-Optical ConvNeXt + 64-Patch Self-Consistency ViT.
    Combines RGB scene, Sobel/Laplacian/shading structural contours, and 8 scale-free physics maps (15 ch)
    to detect synthetic lighting, bokeh/DoF, anatomy/texture shading, and patch-to-patch optical anomalies
    that survive heavy WhatsApp / Instagram / Reddit / WebP internet laundering.
    """

    def __init__(self, in_ch: int = 15, embed_dim: int = 224, dropout: float = 0.25) -> None:
        super().__init__()
        sx = torch.tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]], dtype=torch.float32) / 4.0
        sy = torch.tensor([[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]], dtype=torch.float32) / 4.0
        lap = torch.tensor([[[[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]]], dtype=torch.float32) / 4.0
        self.register_buffer("sobel_x", sx)
        self.register_buffer("sobel_y", sy)
        self.register_buffer("laplacian", lap)

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 80, kernel_size=5, stride=4, padding=2, bias=False),
            nn.GroupNorm(8, 80),
            nn.SiLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            ConvNeXtForensicBlock(80, drop_path=0.04),
            ConvNeXtForensicBlock(80, drop_path=0.05),
            nn.Conv2d(80, 160, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 160),
            nn.SiLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            ConvNeXtForensicBlock(160, drop_path=0.08),
            ConvNeXtForensicBlock(160, drop_path=0.08),
            ConvNeXtForensicBlock(160, drop_path=0.10),
            nn.Conv2d(160, 240, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 240),
            nn.SiLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            ConvNeXtForensicBlock(240, drop_path=0.12),
            ConvNeXtForensicBlock(240, drop_path=0.12),
            nn.Conv2d(240, 240, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, 240),
            nn.SiLU(inplace=True),
        )
        self.to_tokens = nn.Sequential(
            nn.Conv2d(240, 240, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(8, 240),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, 64, 240) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=240,
            nhead=6,
            dim_feedforward=480,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=3, enable_nested_tensor=False)

        # 240 (conv avg) + 240 (tok mean) + 240 (tok std) + 2 (cosine dispersion mean & max) = 722
        self.proj = nn.Sequential(
            nn.LayerNorm(722),
            nn.Dropout(dropout),
            nn.Linear(722, 320, bias=False),
            nn.LayerNorm(320),
            nn.SiLU(inplace=True),
            nn.Linear(320, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Linear(embed_dim, 1)

    @torch.no_grad()
    def _extract_meso_structure(self, x_global: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
        xf = x_global.to(torch.float32)
        gray = 0.299 * xf[:, 0:1] + 0.587 * xf[:, 1:2] + 0.114 * xf[:, 2:3]
        gx = torch.tanh(F.conv2d(gray, self.sobel_x, padding=1) * 2.5)
        gy = torch.tanh(F.conv2d(gray, self.sobel_y, padding=1) * 2.5)
        lap = torch.tanh(F.conv2d(gray, self.laplacian, padding=1) * 3.0)
        mean9 = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        local_contrast = torch.tanh((gray - mean9) * 4.0)
        return torch.cat([gx, gy, lap, local_contrast], dim=1).to(target_dtype)

    def forward(
        self, x_global: torch.Tensor, physics_maps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = physics_maps.dtype
        struct_4ch = self._extract_meso_structure(x_global, dtype)
        x_in = torch.cat([x_global.to(dtype), struct_4ch, physics_maps], dim=1)

        x = self.stem(x_in)
        x = self.stage1(x)
        x = self.stage2(x)
        f16 = self.stage3(x)  # (B, 240, 16, 16)
        conv_avg = F.adaptive_avg_pool2d(f16, 1).flatten(1)

        tok_grid = self.to_tokens(f16)  # (B, 240, 8, 8)
        tokens = tok_grid.flatten(2).transpose(1, 2) + self.pos_embed.to(tok_grid.dtype)

        dev_type = "cuda" if x_global.is_cuda else "cpu"
        with torch.amp.autocast(device_type=dev_type, enabled=False):
            att = self.transformer(tokens.to(torch.float32))  # (B, 64, 240)
            mean_tok = att.mean(dim=1)
            std_tok = att.std(dim=1)

            att_norm = F.normalize(att, dim=-1)
            mean_norm = F.normalize(mean_tok.unsqueeze(1), dim=-1)
            patch_anomaly = 1.0 - (att_norm * mean_norm).sum(dim=-1)  # (B, 64)
            disp_mean = patch_anomaly.mean(dim=1, keepdim=True)
            disp_max = patch_anomaly.max(dim=1, keepdim=True).values
            anomaly_map_8x8 = patch_anomaly.view(-1, 1, 8, 8)

        combined = torch.cat(
            [
                conv_avg.to(dtype),
                mean_tok.to(dtype),
                std_tok.to(dtype),
                disp_mean.to(dtype),
                disp_max.to(dtype),
            ],
            dim=1,
        )
        emb = self.proj(combined)
        return self.head(emb), emb, anomaly_map_8x8.to(dtype)


class DifferentialGramPhysicsStream(nn.Module):
    """
    Stream 4: Intra-Image Differential 2nd-Order Gram Covariance + 28 Scale-Free Physics Ratios.
    """

    def __init__(self, phys_dim: int = 28, embed_dim: int = 224, dropout: float = 0.25) -> None:
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2, dilation=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
        )
        triu_idx = torch.triu_indices(32, 32)
        self.register_buffer("triu_r", triu_idx[0])
        self.register_buffer("triu_c", triu_idx[1])

        in_total = 528 * 3 + phys_dim  # 1584 + 28 = 1612
        self.gram_mlp = nn.Sequential(
            nn.LayerNorm(in_total),
            nn.Dropout(dropout),
            nn.Linear(in_total, 448, bias=False),
            nn.LayerNorm(448),
            nn.SiLU(inplace=True),
        )
        self.res_block = nn.Sequential(
            nn.Dropout(dropout * 0.6),
            nn.Linear(448, 448, bias=False),
            nn.LayerNorm(448),
            nn.SiLU(inplace=True),
            nn.Linear(448, 448, bias=False),
            nn.LayerNorm(448),
        )
        self.out_proj = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Linear(448, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Linear(embed_dim, 1)

    @staticmethod
    def _prep_9ch(x: torch.Tensor) -> torch.Tensor:
        hp = (x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)) * 3.0
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        cdiff = torch.cat([r - g, g - b, b - r], dim=1) * 2.0
        return torch.cat([x, hp, cdiff], dim=1)

    def _compute_gram(self, feat: torch.Tensor) -> torch.Tensor:
        target_dtype = feat.dtype
        dev_type = "cuda" if feat.is_cuda else "cpu"
        with torch.amp.autocast(device_type=dev_type, enabled=False):
            b, c, h, w = feat.shape
            f = feat.to(torch.float32).view(b, c, h * w)
            f = f - f.mean(dim=-1, keepdim=True)
            cov = torch.matmul(f, f.transpose(-1, -2)) / max(1.0, float(h * w - 1))
            cov_signed = torch.sign(cov) * torch.sqrt(torch.abs(cov) + 1e-6)
            upper = cov_signed[:, self.triu_r, self.triu_c]
            upper = torch.clamp(torch.nan_to_num(upper, nan=0.0), -3.0, 3.0)
        return upper.to(target_dtype)

    def forward(
        self,
        x_global: torch.Tensor,
        x_rich: torch.Tensor,
        x_poor: torch.Tensor,
        physics_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = physics_vec.dtype
        b = x_global.shape[0]
        all_in = self._prep_9ch(torch.cat([x_rich.to(dtype), x_poor.to(dtype), x_global.to(dtype)], dim=0))
        feats = self.extractor(all_in)
        f_r, f_p, f_g = feats[:b], feats[b : 2 * b], feats[2 * b :]

        g_r = self._compute_gram(f_r)
        g_p = self._compute_gram(f_p)
        g_g = self._compute_gram(f_g)

        combined = torch.cat([g_r, g_r - g_p, g_g - g_p, physics_vec.to(g_r.dtype)], dim=1)
        h = self.gram_mlp(combined)
        h = h + self.res_block(h)
        emb = self.out_proj(h)
        return self.head(emb), emb


class ArtifactNetV5(nn.Module):
    """
    ArtifactNet v5: 4-Stream Compression-Resilient Forensic Network + 16-Prototype Real-Camera Manifold
    + 12-Family Multi-Generator Attribution Head (100% From Scratch, Zero Pretrained Weights, Zero BatchNorm).
    """

    NUM_PROTOTYPES = 16
    MANIFOLD_DIM = 256
    PHYS_DIM = 28
    NUM_FAMILIES = len(FAMILY_NAMES)

    def __init__(
        self,
        dropout: float = 0.25,
        expert_drop_prob: float = 0.15,
        spectral_drop_prob: float = 0.45,
    ) -> None:
        super().__init__()
        self.expert_drop_prob = expert_drop_prob
        self.embed_dim = 224

        self.spectral_dropout = SpectralRingDropout(img_size=256, prob=spectral_drop_prob)
        self.physics_extractor = ScaleFreePhysicsExtractor(img_size=256)

        self.stream1_micro = MicroResidualStream(embed_dim=224, dropout=dropout)
        self.stream2_wavelet = WaveletPhaseSpectralStream(embed_dim=224, dropout=dropout)
        self.stream3_global = GlobalConsistencyStream(in_ch=15, embed_dim=224, dropout=dropout)
        self.stream4_gram_phys = DifferentialGramPhysicsStream(
            phys_dim=self.PHYS_DIM, embed_dim=224, dropout=dropout
        )

        stream_tf_layer = nn.TransformerEncoderLayer(
            d_model=224,
            nhead=4,
            dim_feedforward=448,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.stream_transformer = nn.TransformerEncoder(stream_tf_layer, num_layers=1, enable_nested_tensor=False)

        self.gate_network = nn.Sequential(
            nn.LayerNorm(224 * 4 + self.PHYS_DIM),
            nn.Linear(224 * 4 + self.PHYS_DIM, 224),
            nn.LayerNorm(224),
            nn.SiLU(inplace=True),
            nn.Linear(224, 4),
        )

        # Projection onto Unit Hypersphere S^255 for 16-Prototype Real-Camera Manifold
        self.manifold_proj = nn.Sequential(
            nn.LayerNorm(224 * 5),
            nn.Dropout(dropout),
            nn.Linear(224 * 5, 384, bias=False),
            nn.LayerNorm(384),
            nn.SiLU(inplace=True),
            nn.Linear(384, self.MANIFOLD_DIM, bias=False),
        )

        # 12-Class Generator Family Attribution Head (0 = Real Camera, 1..11 = Synthetic Families)
        self.family_head = nn.Sequential(
            nn.LayerNorm(self.MANIFOLD_DIM + 224),
            nn.Dropout(dropout * 0.6),
            nn.Linear(self.MANIFOLD_DIM + 224, 224),
            nn.LayerNorm(224),
            nn.SiLU(inplace=True),
            nn.Linear(224, self.NUM_FAMILIES),
        )

        # Binary Discriminative Margin Head
        self.disc_head = nn.Sequential(
            nn.LayerNorm(self.MANIFOLD_DIM + 224 + 4),
            nn.Dropout(dropout * 0.6),
            nn.Linear(self.MANIFOLD_DIM + 224 + 4, 224),
            nn.LayerNorm(224),
            nn.SiLU(inplace=True),
            nn.Linear(224, 1),
        )

        # K=16 Orthogonal Real-Camera Prototypes on S^255
        init_mat = torch.randn(self.MANIFOLD_DIM, self.NUM_PROTOTYPES, dtype=torch.float32)
        q_ortho, _ = torch.linalg.qr(init_mat)
        protos_init = F.normalize(q_ortho[:, : self.NUM_PROTOTYPES].T.contiguous(), dim=-1)

        self.register_buffer("real_prototypes", protos_init)
        self.register_buffer(
            "prototype_usage", torch.full((self.NUM_PROTOTYPES,), 1.0 / self.NUM_PROTOTYPES, dtype=torch.float32)
        )
        self.register_buffer("prototypes_initialized", torch.tensor(False, dtype=torch.bool))

        # Learnable Manifold Boundary Parameters
        self.manifold_tau = nn.Parameter(torch.tensor(0.32, dtype=torch.float32))
        self.manifold_log_scale = nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))

        # Post-Training Multi-Condition Calibration Buffers
        self.register_buffer("blend_alpha", torch.tensor(0.32, dtype=torch.float32))
        self.register_buffer("calib_temp", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("calib_bias", torch.tensor(0.0, dtype=torch.float32))

        # Data-Learned Non-CNN Statistical Physics Weights
        self.register_buffer("stat_weights", torch.zeros(self.PHYS_DIM, dtype=torch.float32))
        self.register_buffer("stat_bias", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("stat_calibrated", torch.tensor(False, dtype=torch.bool))

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
    def _update_real_prototypes(self, z_unit: torch.Tensor, targets: torch.Tensor) -> None:
        real_mask = targets.view(-1) < 0.5
        n_real = int(real_mask.sum().item())
        if n_real < 2:
            return

        z_real = F.normalize(z_unit[real_mask].detach().to(torch.float32), dim=-1)
        protos = F.normalize(self.real_prototypes, dim=-1)

        if not bool(self.prototypes_initialized.item()) and n_real >= self.NUM_PROTOTYPES:
            chosen = [0]
            for _ in range(1, self.NUM_PROTOTYPES):
                sel = z_real[chosen]
                max_sim = torch.matmul(z_real, sel.T).max(dim=1).values
                next_idx = int(torch.argmin(max_sim).item())
                chosen.append(next_idx)
            self.real_prototypes.copy_(F.normalize(z_real[chosen], dim=-1))
            self.prototypes_initialized.fill_(True)
            return

        sim = torch.matmul(z_real, protos.T)
        assign = torch.argmax(sim, dim=1)

        batch_counts = torch.bincount(assign, minlength=self.NUM_PROTOTYPES).to(torch.float32)
        batch_freq = batch_counts / max(1.0, float(n_real))
        self.prototype_usage.mul_(0.98).add_(batch_freq, alpha=0.02)

        for k in range(self.NUM_PROTOTYPES):
            k_mask = assign == k
            if torch.any(k_mask):
                mean_vec = F.normalize(z_real[k_mask].mean(dim=0), dim=-1)
                # Keep subtle inter-prototype repulsion against closest sibling prototype so modes stay diverse
                other_mask = torch.arange(self.NUM_PROTOTYPES, device=protos.device) != k
                max_other_sim, nearest_other = torch.matmul(mean_vec.unsqueeze(0), protos[other_mask].T).squeeze(0).max(dim=0)
                if max_other_sim.item() > 0.72:
                    sibling = protos[other_mask][nearest_other]
                    mean_vec = F.normalize(mean_vec - 0.15 * sibling, dim=-1)
                updated = F.normalize(0.975 * protos[k] + 0.025 * mean_vec, dim=-1)
                self.real_prototypes[k].copy_(updated)
            elif self.prototype_usage[k].item() < 0.012 and n_real >= 4:
                min_max_sim_idx = int(torch.argmin(sim.max(dim=1).values).item())
                revived = F.normalize(0.80 * protos[k] + 0.20 * z_real[min_max_sim_idx], dim=-1)
                self.real_prototypes[k].copy_(revived)
                self.prototype_usage[k].fill_(0.04)

    @staticmethod
    def _auto_split_rich_poor_gpu(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        k = max(2, n_patches // 3)
        rich_idx = sort_idx[:, :k]
        poor_idx = sort_idx[:, -k:]

        repeats = math.ceil(n_patches / k)
        rich_full = rich_idx.repeat(1, repeats)[:, :n_patches].view(b, n_patches, 1, 1, 1).expand(-1, -1, c, ps, ps)
        poor_full = poor_idx.repeat(1, repeats)[:, :n_patches].view(b, n_patches, 1, 1, 1).expand(-1, -1, c, ps, ps)

        x_rich = (
            torch.gather(patches, 1, rich_full)
            .view(b, nh, nw, c, ps, ps)
            .permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view(b, c, nh * ps, nw * ps)
        )
        x_poor = (
            torch.gather(patches, 1, poor_full)
            .view(b, nh, nw, c, ps, ps)
            .permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view(b, c, nh * ps, nw * ps)
        )
        return x_rich, x_poor

    def forward(
        self,
        x_rich: torch.Tensor,
        x_poor: torch.Tensor | None = None,
        x_global: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        paired_batch: bool = False,
        return_details: bool = False,
        return_maps: bool = False,
    ):
        if x_rich.dtype == torch.uint8:
            x_rich = x_rich.to(torch.float32) / 255.0
        if x_poor is not None and x_poor.dtype == torch.uint8:
            x_poor = x_poor.to(torch.float32) / 255.0
        if x_global is not None and x_global.dtype == torch.uint8:
            x_global = x_global.to(torch.float32) / 255.0

        if x_poor is None and x_global is None:
            raw_img = x_rich
            x_global = F.interpolate(raw_img, size=(256, 256), mode="area")
            x_rich, x_poor = self._auto_split_rich_poor_gpu(raw_img)
        elif x_poor is None:
            x_rich, x_poor = self._auto_split_rich_poor_gpu(x_rich)
        elif x_global is None:
            x_global = x_rich

        if self.training:
            x_global, x_rich, x_poor = self.spectral_dropout(
                x_global, x_rich, x_poor, paired_batch=paired_batch
            )

        physics_maps, physics_vec = self.physics_extractor(x_global, x_rich, x_poor)
        dtype = physics_maps.dtype

        logit1, emb1, f4_rich = self.stream1_micro(x_rich.to(dtype), x_poor.to(dtype))
        logit2, emb2 = self.stream2_wavelet(x_global.to(dtype), x_rich.to(dtype))
        logit3, emb3, anomaly_8x8 = self.stream3_global(x_global.to(dtype), physics_maps)
        logit4, emb4 = self.stream4_gram_phys(
            x_global.to(dtype), x_rich.to(dtype), x_poor.to(dtype), physics_vec
        )

        stacked = torch.stack([emb1, emb2, emb3, emb4], dim=1)  # (B, 4, 224)
        dev_type = "cuda" if x_global.is_cuda else "cpu"
        with torch.amp.autocast(device_type=dev_type, enabled=False):
            refined = self.stream_transformer(stacked.to(torch.float32)).to(dtype)

        flat_embs = refined.flatten(1)  # (B, 896)
        gate_logits = self.gate_network(torch.cat([flat_embs, physics_vec.to(flat_embs.dtype)], dim=1))

        if self.training and self.expert_drop_prob > 0.0:
            keep = torch.empty_like(gate_logits).bernoulli_(1.0 - self.expert_drop_prob)
            too_few = keep.sum(dim=1, keepdim=True) < 2
            keep = torch.where(too_few, torch.ones_like(keep), keep)
            gate_logits = gate_logits.masked_fill(keep == 0, -1e4)

        gate_weights = F.softmax(gate_logits, dim=1)
        weighted_emb = (refined * gate_weights.unsqueeze(-1)).sum(dim=1)  # (B, 224)

        z_raw = self.manifold_proj(torch.cat([flat_embs, weighted_emb], dim=1))
        family_in = torch.cat([F.normalize(z_raw, dim=-1), weighted_emb], dim=1)
        family_logits = self.family_head(family_in)

        with torch.amp.autocast(device_type=dev_type, enabled=False):
            z_unit = F.normalize(z_raw.to(torch.float32), dim=-1)  # (B, 256) on S^255
            if self.training and targets is not None:
                self._update_real_prototypes(z_unit, targets)

            protos = F.normalize(self.real_prototypes, dim=-1)  # (K, 256)
            cos_sim = torch.matmul(z_unit, protos.T)  # (B, K) in [-1, 1]
            cos_dists = 1.0 - cos_sim  # (B, K) in [0, 2]

            # Exact min distance + non-negative Log-Mean-Exp smooth minimum (never negative!)
            d_min = cos_dists.min(dim=1, keepdim=True).values
            if self.training:
                t_sm = 0.07
                lme = torch.logsumexp(-cos_dists / t_sm, dim=1, keepdim=True) - math.log(self.NUM_PROTOTYPES)
                d_lme = torch.clamp(-t_sm * lme, min=0.0)
                d_real = 0.5 * d_min + 0.5 * d_lme
            else:
                d_real = d_min

            tau = torch.clamp(self.manifold_tau, 0.01, 0.85)
            m_scale = torch.clamp(torch.exp(self.manifold_log_scale), 2.5, 60.0)
            manifold_logit = m_scale * (d_real - tau)

            # Log-odds of "Any AI Family (1..11)" vs "Real Camera (0)" from 12-family head
            fam_f32 = family_logits.to(torch.float32)
            family_ai_logit = torch.logsumexp(fam_f32[:, 1:], dim=1, keepdim=True) - fam_f32[:, 0:1]

        stream_logits = torch.cat([logit1, logit2, logit3, logit4], dim=1)
        weighted_stream_logit = (stream_logits * gate_weights).sum(dim=1, keepdim=True)

        disc_in = torch.cat([z_unit.to(weighted_emb.dtype), weighted_emb, stream_logits], dim=1)
        disc_head_logit = self.disc_head(disc_in)
        disc_logit = (
            0.50 * disc_head_logit.to(torch.float32)
            + 0.30 * family_ai_logit
            + 0.20 * weighted_stream_logit.to(torch.float32)
        )

        with torch.amp.autocast(device_type=dev_type, enabled=False):
            alpha = torch.clamp(self.blend_alpha, 0.15, 0.75)
            raw_fused_logit = (1.0 - alpha) * disc_logit + alpha * manifold_logit
            temp = torch.clamp(self.calib_temp, 0.30, 4.5)
            final_logit = (raw_fused_logit + self.calib_bias) / temp

            stat_logit = (
                torch.matmul(physics_vec.to(torch.float32), self.stat_weights.unsqueeze(1)) + self.stat_bias
            )

        out_logit = final_logit.to(dtype)

        if return_details:
            details = {
                "logit1_micro": logit1,
                "logit2_wavelet": logit2,
                "logit3_global": logit3,
                "logit4_gram_phys": logit4,
                "family_logits": family_logits,
                "family_ai_logit": family_ai_logit.to(dtype),
                "disc_logit": disc_logit.to(dtype),
                "manifold_logit": manifold_logit.to(dtype),
                "raw_fused_logit": raw_fused_logit.to(dtype),
                "stat_logit": stat_logit.to(dtype),
                "stat_calibrated": bool(self.stat_calibrated.item()),
                "d_real": d_min,
                "d_real_train": d_real,
                "manifold_tau": tau.detach(),
                "z_unit": z_unit,
                "gate_weights": gate_weights,
                "physics_vec": physics_vec.to(torch.float32),
            }
            if return_maps:
                details["anomaly_8x8"] = anomaly_8x8.to(torch.float32)
                details["physics_maps"] = physics_maps.to(torch.float32)
                details["f4_rich"] = f4_rich.to(torch.float32)
            return out_logit, details

        return out_logit


# Clean aliases for compatibility across scripts & exports
ArtifactNet = ArtifactNetV5
ArtifactNetV2 = ArtifactNetV5
ArtifactNetV4 = ArtifactNetV5
