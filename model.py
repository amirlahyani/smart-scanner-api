import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction, bias=False),
            nn.ReLU(True),
            nn.Linear(ch // reduction, ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(
            self.avg(x).squeeze(-1).squeeze(-1)
        ).unsqueeze(-1).unsqueeze(-1)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch)
        )
        self.ca = ChannelAttention(ch)

    def forward(self, x):
        return x + self.ca(self.body(x))


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_res=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
            *[ResBlock(out_ch) for _ in range(n_res)]
        )
        self.down = nn.Conv2d(out_ch, out_ch, 2, stride=2, bias=False)

    def forward(self, x):
        s = self.conv(x)
        return self.down(s), s


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, n_res=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
            *[ResBlock(out_ch) for _ in range(n_res)]
        )

    def forward(self, x, s):
        x = self.up(x)
        if x.shape != s.shape:
            x = F.interpolate(x, size=s.shape[2:],
                              mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, s], dim=1))


class DocumentDenoiser(nn.Module):
    def __init__(self, base_ch=32):
        super().__init__()
        b = base_ch
        self.enc1 = EncoderBlock(3, b, n_res=2)
        self.enc2 = EncoderBlock(b, b * 2, n_res=2)
        self.enc3 = EncoderBlock(b * 2, b * 4, n_res=3)
        self.enc4 = EncoderBlock(b * 4, b * 8, n_res=3)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(b * 8, b * 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(b * 16),
            nn.ReLU(True),
            ResBlock(b * 16),
            ResBlock(b * 16),
            ChannelAttention(b * 16)
        )
        self.dec4 = DecoderBlock(b * 16, b * 8, b * 8, n_res=3)
        self.dec3 = DecoderBlock(b * 8, b * 4, b * 4, n_res=3)
        self.dec2 = DecoderBlock(b * 4, b * 2, b * 2, n_res=2)
        self.dec1 = DecoderBlock(b * 2, b, b, n_res=2)
        self.head = nn.Conv2d(b, 3, 1, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)
        bn = self.bottleneck(x4)
        d4 = self.dec4(bn, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return torch.clamp(x - self.head(d1), -1.0, 1.0)