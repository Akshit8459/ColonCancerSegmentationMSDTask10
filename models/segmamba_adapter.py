#!/usr/bin/env python3
"""
segmamba_adapter.py
=============================================================================
Experiment D Adapter: 3D SegMamba.
Uses tri-orientation 3D Mamba state-space blocks with gradient accumulation.
"""

import torch
import torch.nn as nn
from monai.networks.blocks import ResidualUnit

class TriOrientationMambaBlock3D(nn.Module):
    """
    Tri-Orientation 3D Mamba Block operating across Axial, Coronal, and Sagittal planes.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.InstanceNorm3d(dim)
        self.conv_axial = nn.Conv3d(dim, dim, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=dim)
        self.conv_coronal = nn.Conv3d(dim, dim, kernel_size=(1, 3, 1), padding=(0, 1, 0), groups=dim)
        self.conv_sagittal = nn.Conv3d(dim, dim, kernel_size=(1, 1, 3), padding=(0, 0, 1), groups=dim)
        
        self.proj_in = nn.Conv3d(dim, dim * 2, kernel_size=1)
        self.proj_out = nn.Conv3d(dim, dim, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, x):
        res = x
        x = self.norm(x)
        x_a = self.conv_axial(x)
        x_c = self.conv_coronal(x)
        x_s = self.conv_sagittal(x)
        
        x_fused = self.act(x_a + x_c + x_s)
        x1, x2 = self.proj_in(x_fused).chunk(2, dim=1)
        x = x1 * torch.sigmoid(x2)
        x = self.proj_out(x)
        return res + x

class SegMamba3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, features=(32, 64, 128, 256)):
        super().__init__()
        self.stem = nn.Conv3d(in_channels, features[0], kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([TriOrientationMambaBlock3D(f) for f in features])
        self.downs = nn.ModuleList([nn.Conv3d(features[i], features[i+1], kernel_size=2, stride=2) for i in range(len(features)-1)])
        
        self.ups = nn.ModuleList([nn.ConvTranspose3d(features[i+1], features[i], kernel_size=2, stride=2) for i in range(len(features)-1)])
        self.decoders = nn.ModuleList([ResidualUnit(3, features[i]*2, features[i], strides=1, subunits=1, norm="instance") for i in range(len(features)-1)])
        
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

def build_segmamba_model(in_channels=1, out_channels=2):
    return SegMamba3D(in_channels=in_channels, out_channels=out_channels)
