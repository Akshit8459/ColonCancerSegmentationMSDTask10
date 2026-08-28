#!/usr/bin/env python3
"""
nnuzoo_adapter.py
=============================================================================
Experiment F Adapter: nnUZoo / X2Net Hybrid Encoder-Swap Architecture.
Combines 3D ConvNeXt / X2Net hybrid blocks with 3D UNet decoder using ConvTranspose3d upsampling.
"""

import torch
import torch.nn as nn
from monai.networks.blocks import ResidualUnit

class ConvNeXtBlock3D(nn.Module):
    """
    3D ConvNeXt / X2Net depthwise block.
    """
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.InstanceNorm3d(dim)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return residual + x

class X2Net3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, features=(32, 64, 128, 256)):
        super().__init__()
        self.stem = nn.Conv3d(in_channels, features[0], kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ConvNeXtBlock3D(f) for f in features])
        self.downs = nn.ModuleList([
            nn.Conv3d(features[i], features[i+1], kernel_size=2, stride=2)
            for i in range(len(features)-1)
        ])
        
        # Transposed convolution for spatial upsampling + channel reduction
        self.ups = nn.ModuleList([
            nn.ConvTranspose3d(features[i+1], features[i], kernel_size=2, stride=2)
            for i in range(len(features)-1)
        ])
        
        # Decoder ResidualUnits: after concatenation, channels = features[i] (from up) + features[i] (skip) = 2*features[i]
        self.decoders = nn.ModuleList([
            ResidualUnit(3, features[i]*2, features[i], strides=1, subunits=1, norm="instance")
            for i in range(len(features)-1)
        ])
        
        self.out_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for i in range(len(self.blocks)-1):
            x = self.blocks[i](x)
            skips.append(x)
            x = self.downs[i](x)
            
        x = self.blocks[-1](x)
        
        for up, dec, skip in zip(reversed(self.ups), reversed(self.decoders), reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
            
        return self.out_conv(x)

def build_nnuzoo_model(in_channels=1, out_channels=2):
    return X2Net3D(in_channels=in_channels, out_channels=out_channels)
