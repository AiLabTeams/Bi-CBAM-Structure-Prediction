import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, ratio: int = 8):
        super().__init__()
        hidden = max(channels // ratio, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        attn = self.sigmoid(avg_out + max_out)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels: int, ratio: int = 8):
        super().__init__()
        self.ca = ChannelAttention(channels, ratio=ratio)
        self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ratio: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNAct(in_ch, out_ch, 3, 1),
            ConvBNAct(out_ch, out_ch, 3, 1),
        )
        self.cbam = CBAM(out_ch, ratio=ratio)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.cbam(x)
        x = self.pool(x)
        return x


class DecoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ratio: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNAct(in_ch, out_ch, 3, 1),
            ConvBNAct(out_ch, out_ch, 3, 1),
        )
        self.cbam = CBAM(out_ch, ratio=ratio)

    def forward(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        x = self.conv(x)
        x = self.cbam(x)
        return x


class InverseBiCBAMNetV2(nn.Module):
    def __init__(self, in_channels: int = 1, ratio: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            EncoderStage(in_channels, 16, ratio=ratio),
            EncoderStage(16, 32, ratio=ratio),
            EncoderStage(32, 64, ratio=ratio),
            EncoderStage(64, 96, ratio=ratio),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(84, 20),
            nn.ReLU(inplace=True),
        )
        self.head_cont = nn.Linear(20, 8)
        self.head_time = nn.Linear(20, 3)

    @staticmethod
    def _normalize_angle_pairs(angle_pairs: torch.Tensor) -> torch.Tensor:
        angle_pairs = torch.tanh(angle_pairs)
        reshaped = angle_pairs.view(angle_pairs.shape[0], 3, 2)
        norm = torch.norm(reshaped, dim=-1, keepdim=True).clamp_min(1e-6)
        reshaped = reshaped / norm
        return reshaped.view(angle_pairs.shape[0], 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        feat = self.gap(feat)
        shared = self.shared(feat)
        raw_cont = self.head_cont(shared)
        times_logits = self.head_time(shared)
        ia_rad = torch.sigmoid(raw_cont[:, :2])
        angle_pairs = self._normalize_angle_pairs(raw_cont[:, 2:])
        return torch.cat([ia_rad, angle_pairs, times_logits], dim=1)


class ForwardBiCBAMNetV2(nn.Module):
    def __init__(self, out_channels: int = 1, img_size: Tuple[int, int] = (138, 80), ratio: int = 8):
        super().__init__()
        self.img_size = img_size
        self.seed_size = self._compute_seed_size(img_size)
        sh, sw = self.seed_size
        self.seed_channels = 96
        self.fc = nn.Sequential(
            nn.Linear(11, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, self.seed_channels * sh * sw),
            nn.ReLU(inplace=True),
        )
        self.dec1 = DecoderStage(96, 64, ratio=ratio)
        self.dec2 = DecoderStage(64, 48, ratio=ratio)
        self.dec3 = DecoderStage(48, 32, ratio=ratio)
        self.dec4 = DecoderStage(32, 16, ratio=ratio)
        self.out_conv = nn.Sequential(
            ConvBNAct(16, 16, 3, 1),
            nn.Conv2d(16, out_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    @staticmethod
    def _compute_seed_size(img_size: Tuple[int, int], levels: int = 4) -> Tuple[int, int]:
        h, w = img_size
        for _ in range(levels):
            h = math.ceil(h / 2)
            w = math.ceil(w / 2)
        return h, w

    def forward(self, packed_params: torch.Tensor) -> torch.Tensor:
        b = packed_params.shape[0]
        sh, sw = self.seed_size
        x = self.fc(packed_params)
        x = x.view(b, self.seed_channels, sh, sw)
        h, w = self.img_size
        sizes = [
            (math.ceil(h / 8), math.ceil(w / 8)),
            (math.ceil(h / 4), math.ceil(w / 4)),
            (math.ceil(h / 2), math.ceil(w / 2)),
            (h, w),
        ]
        x = self.dec1(x, sizes[0])
        x = self.dec2(x, sizes[1])
        x = self.dec3(x, sizes[2])
        x = self.dec4(x, sizes[3])
        return self.out_conv(x)
