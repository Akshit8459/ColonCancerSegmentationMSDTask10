#!/usr/bin/env python3
"""
umamba_adapter.py
=============================================================================
Experiment B Adapter: 3D U-Mamba (No Pretraining).
Uses a 3D U-Mamba encoder architecture with Conv-Mamba state-space blocks
and a 3D UNet decoder.
"""

import torch
import torch.nn as nn
from monai.networks.blocks import ResidualUnit, UpSample

class MambaBlock3D(nn.Module):
    """
    3D Mamba / State-Space inspired Block: Depthwise Conv3d + Gated Linear Units + Channel Projection.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.InstanceNorm3d(dim)
        self.conv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.proj_in = nn.Conv3d(dim, dim * 2, kernel_size=1)
        self.proj_out = nn.Conv3d(dim, dim, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.conv(x)
        x = self.act(x)
        x1, x2 = self.proj_in(x).chunk(2, dim=1)
        x = x1 * torch.sigmoid(x2)
        x = self.proj_out(x)
        return residual + x

class UMamba3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, features=(32, 64, 128, 256)):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.mamba_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        curr_in = in_channels
        for feat in features:
            self.encoders.append(ResidualUnit(3, curr_in, feat, strides=1, num_res_units=2, norm="instance"))
            self.mamba_blocks.append(MambaBlock3D(feat))
            self.downs.append(nn.Conv3d(feat, feat, kernel_size=2, stride=2))
            curr_in = feat

        self.bottleneck = nn.Sequential(
            ResidualUnit(3, features[-1], features[-1] * 2, strides=1, num_res_units=2, norm="instance"),
            MambaBlock3D(features[-1] * 2)
        )

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        rev_features = list(reversed(features))
        curr_in = features[-1] * 2
        for feat in rev_features:
            self.ups.append(UpSample(3, curr_in, feat, scale_factor=2, mode="nontrainable"))
            self.decoders.append(ResidualUnit(3, feat * 2, feat, strides=1, num_res_units=2, norm="instance"))
            curr_in = feat

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc, mamba, down in zip(self.encoders, self.mamba_blocks, self.downs):
            x = enc(x)
            x = mamba(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.final_conv(x)

def build_umamba_model(in_channels=1, out_channels=2):
    return UMamba3D(in_channels=in_channels, out_channels=out_channels)
