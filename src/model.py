import torch
import torch.nn as nn
import torch.nn.functional as F


class HighPassExtractor(nn.Module):
    """
    ბლოკი 1: სიხშირული (High-Pass) ფილტრი.
    სურათიდან აქრობს გლუვ ფერებს და ტოვებს მხოლოდ მიკრო-ტექსტურებს,
    კიდეებს და AI გენერატორისთვის დამახასიათებელ პიქსელურ ხმაურს.
    """
    def __init__(self):
        super().__init__()
        # 3x3 Laplacian-ის მსგავსი ფილტრი მიკრო-არტეფაქტების გამოსავლენად
        kernel = torch.tensor([
            [-1.0, -1.0, -1.0],
            [-1.0,  8.0, -1.0],
            [-1.0, -1.0, -1.0]
        ], dtype=torch.float32) / 8.0

        # ფილტრს ვამრავლებთ სამივე (R, G, B) არხისთვის: ფორმა (3, 1, 3, 3)
        kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        # register_buffer ნიშნავს, რომ ეს ფიქსირებული მათემატიკური ფილტრია და GPU-ზე ცხოვრობს
        self.register_buffer("kernel", kernel)

        # დამატებითი სწავლებადი 3x3 კონვოლუცია სიხშირული არხების დასახვეწად
        self.learnable_freq = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # groups=3 უზრუნველყოფს, რომ R, G და B არხები ცალ-ცალკე გაიფილტროს
        high_pass = F.conv2d(x, self.kernel.to(x.dtype), padding=1, groups=3)
        return self.learnable_freq(high_pass)


class SEAttention(nn.Module):
    """
    ბლოკი 4: Squeeze-and-Excitation (SE) ყურადღების მექანიზმი.
    ასწავლის ქსელს, რომელ ფილტრებს (არხებს) მიაქციოს მეტი ყურადღება.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced_channels = max(channels // reduction, 8)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, reduced_channels, bias=False),
            nn.GELU(),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # ვკუმშავთ თითოეულ არხს 1 რიცხვამდე და ვითვლით მის მნიშვნელობას (0-დან 1-მდე)
        weights = self.excitation(self.squeeze(x)).view(b, c, 1, 1)
        # ვამრავლებთ არხებს თავიანთ წონებზე
        return x * weights


class ResidualAttentionBlock(nn.Module):
    """
    ბლოკი 2 და 3: ჩვენი მთავარი სამშენებლო აგური.
    აერთიანებს კონვოლუციებს, Depthwise ფილტრაციას, SE Attention-ს და Skip Connection-ს.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, drop_rate: float = 0.0):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.se = SEAttention(out_channels, reduction=16)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(p=drop_rate) if drop_rate > 0 else nn.Identity()

        # თუ ზომა ან არხების რაოდენობა იცვლება, Skip Connection-საც ვადაპტირებთ 1x1 კონვოლუციით
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.se(out)

        # Skip Connection: ვუმატებთ საწყის სიგნალს
        out = out + identity
        return self.act(out)


class ArtifactNet(nn.Module):
    """
    სრული ნეირონული ქსელი ნულიდან (From Scratch) AI vs Real სურათების კლასიფიკაციისთვის.
    """
    def __init__(self, num_classes: int = 1, dropout_rate: float = 0.3):
        super().__init__()

        # 1. Dual-Stream Stem: RGB ნაკადი + სიხშირული (High-Pass) ნაკადი
        self.freq_extractor = HighPassExtractor()  # გამოაქვს 16 არხი
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU()
        )
        # 48 (RGB) + 16 (Freq) = 64 არხი. ვაპატარავებთ ზომას 2-ჯერ საწყის ეტაპზე
        self.stem_downsample = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        # 2. ოთხი იერარქიული საფეხური (Stages)
        # Stage 1: 64 არხი (მაღალი რეზოლუცია, მიკრო-დეტალები)
        self.stage1 = nn.Sequential(
            ResidualAttentionBlock(64, 64, stride=1),
            ResidualAttentionBlock(64, 64, stride=1)
        )

        # Stage 2: 128 არხი (ტექსტურები და ლოკალური პატერნები)
        self.stage2 = nn.Sequential(
            ResidualAttentionBlock(64, 128, stride=2, drop_rate=0.05),
            ResidualAttentionBlock(128, 128, stride=1, drop_rate=0.05),
            ResidualAttentionBlock(128, 128, stride=1, drop_rate=0.05)
        )

        # Stage 3: 256 არხი (ობიექტის ნაწილები, ჩრდილები, ანატომია)
        self.stage3 = nn.Sequential(
            ResidualAttentionBlock(128, 256, stride=2, drop_rate=0.1),
            ResidualAttentionBlock(256, 256, stride=1, drop_rate=0.1),
            ResidualAttentionBlock(256, 256, stride=1, drop_rate=0.1),
            ResidualAttentionBlock(256, 256, stride=1, drop_rate=0.1)
        )

        # Stage 4: 512 არხი (გლობალური კონტექსტი და სემანტიკა)
        self.stage4 = nn.Sequential(
            ResidualAttentionBlock(256, 512, stride=2, drop_rate=0.15),
            ResidualAttentionBlock(512, 512, stride=1, drop_rate=0.15),
            ResidualAttentionBlock(512, 512, stride=1, drop_rate=0.15)
        )

        # 3. საბოლოო კლასიფიკატორი (Dual Pooling: Avg + Max)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(512 * 2),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512 * 2, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes)
        )

        # წონების სწორი ინიციალიზაცია (Kaiming He Initialization)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # ცალკე გამოვყოფთ ფუნქციას, რომ მოგვიანებით Grad-CAM (სითბური რუკა) მარტივად დავხატოთ
        rgb_feat = self.rgb_stem(x)
        freq_feat = self.freq_extractor(x)

        # ვაერთიანებთ ფერსა და სიხშირულ არხებს (48 + 16 = 64)
        x = torch.cat([rgb_feat, freq_feat], dim=1)
        x = self.stem_downsample(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        # ვაერთიანებთ საშუალო და მაქსიმალურ სიგნალებს (512 + 512 = 1024)
        pooled = torch.cat([self.avg_pool(features), self.max_pool(features)], dim=1)
        logits = self.classifier(pooled)
        return logits