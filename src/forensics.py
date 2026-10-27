import torch
import torch.nn as nn
import torch.nn.functional as F


class NPRLayer(nn.Module):
    """
    Neighboring Pixel Relationships (NPR - CVPR 2024 ადაპტაცია).
    აშლის სემანტიკურ შინაარსს და ააშკარავებს გენერატორების Up-sampling არტეფაქტებს
    ლოკალურ 2x2 და 3x3 სამეზობლოში.
    """
    def __init__(self):
        super().__init__()
        # 4 მიმართულებითი მეზობელი პიქსელის სხვაობის ფილტრი (ჰორიზონტალური, ვერტიკალური, 2 დიაგონალი)
        kernels = torch.tensor([
            [[[0, 0, 0], [0, 1, -1], [0, 0, 0]]],
            [[[0, 0, 0], [0, 1, 0], [0, -1, 0]]],
            [[[0, 0, 0], [0, 1, 0], [0, 0, -1]]],
            [[[0, 0, -1], [0, 1, 0], [0, 0, 0]]],
        ], dtype=torch.float32)  # (4, 1, 3, 3)

        # ვამრავლებთ 3-ვე RGB არხისთვის (Depthwise კონვოლუციისთვის: 12, 1, 3, 3)
        self.register_buffer("npr_kernels", kernels.repeat(3, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. მიმართულებითი NPR (12 არხი)
        directional_npr = F.conv2d(x, self.npr_kernels.to(x.dtype), padding=1, groups=3)

        # 2. Local Down-Up Interpolation Residual (3 არხი)
        # ვამოწმებთ, რამდენად ჰგავს სურათი უკვე ინტერპოლირებულ (Up-sampled) ბადეს
        down = F.avg_pool2d(x, kernel_size=2, stride=2)
        up = F.interpolate(down, size=x.shape[-2:], mode="bilinear", align_corners=False)
        interp_residual = x - up

        # ვაბრუნებთ 15 არხს და ვუკეთებთ რბილ ტრუნკაციას (Tanh), რომ ძლიერმა კიდეებმა არ გადაფაროს მიკრო-ხმაური
        out = torch.cat([directional_npr, interp_residual], dim=1)
        return torch.tanh(out * 3.0)


class SRMFilterBank(nn.Module):
    """
    Spatial Rich Models (SRM) ფილტრების ბანკი ციფრული ფორენზიკიდან.
    იჭერს მე-2 და მე-3 რიგის მიკრო-ტექსტურულ ანომალიებს.
    """
    def __init__(self):
        super().__init__()
        # 6 ფუნდამენტური SRM ფილტრი (Edge, Square 3x3, KV-filter, 2nd-order H/V/Diag)
        srm_1 = [[0, 0, 0], [-1, 2, -1], [0, 0, 0]]          # 2nd order Horizontal
        srm_2 = [[0, -1, 0], [0, 2, 0], [0, -1, 0]]          # 2nd order Vertical
        srm_3 = [[-1, 0, 0], [0, 2, 0], [0, 0, -1]]          # 2nd order Diagonal
        srm_4 = [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]       # 3x3 Square Edge (KB kernel)
        srm_5 = [[-1, 2, -2, 2, -1],
                 [2, -6, 8, -6, 2],
                 [-2, 8, -12, 8, -2],
                 [2, -6, 8, -6, 2],
                 [-1, 2, -2, 2, -1]]                         # 5x5 KV High-order kernel

        k3 = torch.tensor([srm_1, srm_2, srm_3, srm_4], dtype=torch.float32).unsqueeze(1)
        k3[0:3] /= 2.0
        k3[3] /= 4.0

        k5 = torch.tensor([srm_5], dtype=torch.float32).unsqueeze(1) / 12.0

        self.register_buffer("weight_3x3", k3.repeat(3, 1, 1, 1))  # (12, 1, 3, 3)
        self.register_buffer("weight_5x5", k5.repeat(3, 1, 1, 1))  # (3, 1, 5, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_3x3 = F.conv2d(x, self.weight_3x3.to(x.dtype), padding=1, groups=3)
        out_5x5 = F.conv2d(x, self.weight_5x5.to(x.dtype), padding=2, groups=3)
        out = torch.cat([out_3x3, out_5x5], dim=1)  # 15 არხი
        # SRM-ის კლასიკური ტრუნკაცია (Clipping), რათა სემანტიკური კონტურები ჩაიხშოს
        return torch.clamp(out, min=-2.0, max=2.0)


class BayarConv2d(nn.Module):
    """
    Constrained Convolutional Layer (Bayar & Stamm, IEEE TIFS).
    სწავლებადი ფორენზიკული შრე, სადაც ცენტრალური წონა ყოველთვის -1-ია,
    ხოლო გარშემო წონების ჯამი = +1. ქსელი ვერასდროს დაინახავს ფერს/განათებას,
    მაგრამ სწავლობს ოპტიმალურ ანტი-კომპრესიულ ფილტრებს.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 12, kernel_size: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.xavier_uniform_(self.weight)

        # ნიღაბი ცენტრალური ელემენტის გამოსაყოფად
        mask = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32)
        mask[:, :, kernel_size // 2, kernel_size // 2] = 0.0
        self.register_buffer("surround_mask", mask)

    def _get_constrained_weights(self) -> torch.Tensor:
        w = self.weight * self.surround_mask
        # გარშემო წონების ნორმალიზაცია ისე, რომ ჯამი იყოს +1 თითოეულ არხზე
        surround_sum = w.sum(dim=(2, 3), keepdim=True) + 1e-7
        w = w / surround_sum
        # ცენტრალურ პიქსელზე ვსვამთ -1 / in_channels (ჯამურად -1)
        center = self.kernel_size // 2
        center_mask = 1.0 - self.surround_mask
        w = w - center_mask * (1.0 / self.in_channels)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constrained_w = self._get_constrained_weights().to(x.dtype)
        return F.conv2d(x, constrained_w, padding=self.padding)


class ChromaResidualExtractor(nn.Module):
    """
    ფერთა არხებს შორის (R-G, G-B, B-R) კორელაციის ანალიზატორი.
    იჭერს 'AI Slop'-ის პლასტმასისებრ ტექსტურას და კამერის Bayer CFA-ს არარსებობას.
    """
    def __init__(self):
        super().__init__()
        laplacian = torch.tensor(
            [[[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]], dtype=torch.float32
        ) / 4.0
        self.register_buffer("laplacian", laplacian.repeat(3, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        rg = r - g
        gb = g - b
        br = b - r
        chroma_diff = torch.cat([rg, gb, br], dim=1)  # (B, 3, H, W)
        chroma_hf = F.conv2d(chroma_diff, self.laplacian.to(x.dtype), padding=1, groups=3)
        return torch.cat([chroma_diff, chroma_hf], dim=1)  # 6 არხი


class ForensicFrontEnd(nn.Module):
    """
    სრული ფორენზიკული შესასვლელი (NPR + SRM + BayarConv + Chroma Residual).
    3-არხიანი RGB სურათიდან ქმნის 48-არხიან სუფთა ფორენზიკულ ტენზორს.
    """
    def __init__(self):
        super().__init__()
        self.npr = NPRLayer()                    # 15 არხი
        self.srm = SRMFilterBank()               # 15 არხი
        self.bayar = BayarConv2d(3, 12, 5)       # 12 არხი
        self.chroma = ChromaResidualExtractor()  # 6 არხი
        self.norm = nn.BatchNorm2d(48)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_npr = self.npr(x)
        f_srm = self.srm(x)
        f_bayar = self.bayar(x)
        f_chroma = self.chroma(x)
        out = torch.cat([f_npr, f_srm, f_bayar, f_chroma], dim=1)  # (B, 48, H, W)
        return self.norm(out)