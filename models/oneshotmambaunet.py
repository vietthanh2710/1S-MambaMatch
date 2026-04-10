from __future__ import annotations

import math
import random
from functools import partial
from typing import Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath

from models.mivss import MIVSSBlock

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def augment_feature_map(x, p=0.1, noise_std=0.05, shuffle_ratio=0.25):
    if random.random() > p:
        return x
    B, C, H, W = x.shape
    x = x + torch.randn_like(x) * noise_std
    num_shuffle = int(C * shuffle_ratio)
    if num_shuffle > 1:
        perm = torch.randperm(num_shuffle, device=x.device)
        shuffled = x[:, :num_shuffle][:, perm]
        x = torch.cat([shuffled, x[:, num_shuffle:]], dim=1)
    return x

class OneShotMambaUnet(nn.Module):
    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 2,
        bilinear: bool = False,
        drop_probs: Tuple[float, ...] = (0.2, 0.2, 0.2, 0.1, 0.1),
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.drop_probs = drop_probs

        factor = 2 if bilinear else 1
        
        self.inc = DoubleConv(n_channels, 16)
        self.downs = nn.ModuleList([
            Down(16, 32),
            Down(32, 64),
            Down(64, 128),
            Down(128, 256 // factor),
        ])

        self.fewshot_module = MIVSSBlock(
            hidden_dim=256,
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_conv=3,
            ssm_conv_bias=True,
            mlp_ratio=4.0,
            gmlp=True,
            post_norm=True,
        )

        self.ups = nn.ModuleList([
            Up(256, 128 // factor, bilinear),
            Up(128, 64 // factor, bilinear),
            Up(64, 32 // factor, bilinear),
            Up(32, 16, bilinear),
        ])

        self.outc = OutConv(16, n_classes)

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = [self.inc(x)]
        for down in self.downs:
            feats.append(down(feats[-1]))
        return feats  # [x1, x2, x3, x4, x5]

    def decode(self, x5, skips: List[torch.Tensor]) -> torch.Tensor:
        x = x5
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)
        return self.outc(x)


    def apply_feature_dropout(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        device = feats[0].device
        bs = feats[0].shape[0]
        half = bs // 2

        dropped_feats = []

        for feat, p in zip(feats, self.drop_probs):
            feat = augment_feature_map(feat, p=p)

            C = feat.shape[1]

            mask1 = (torch.rand(half, C, device=device) > 0.5).float() * 2.0
            mask2 = (torch.rand(half, C, device=device) > 0.5).float() * 2.0

            keep_ratio = 0.5
            num_kept = int(half * keep_ratio)

            keep_idx1 = torch.randperm(half, device=device)[:num_kept]
            keep_idx2 = torch.randperm(half, device=device)[:num_kept]

            mask1[keep_idx1] = 1.0
            mask2[keep_idx2] = 1.0

            mask = torch.cat([mask1, mask2], dim=0)
            mask = mask.unsqueeze(-1).unsqueeze(-1)

            dropped_feats.append(feat * mask)

        return dropped_feats

    def forward(self, x, s, s_msk, comp_drop: bool = False):

        q_feats = self.encode(x)

        if comp_drop:
            q_feats = self.apply_feature_dropout(q_feats)

            q_ft_out, _ = self.fewshot_module(q_feats[-1], s, s_msk)
            q_ft_out = q_ft_out.permute(0, 3, 1, 2)

            logits = self.decode(q_ft_out, q_feats[:-1])
            return logits

        s_feats = self.encode(s)

        q_ft_out, _ = self.fewshot_module(q_feats[-1], s_feats[-1], s_msk)
        q_ft_out = q_ft_out.permute(0, 3, 1, 2)

        logits = self.decode(q_ft_out, q_feats[:-1])

        return logits, s_feats[-1]

