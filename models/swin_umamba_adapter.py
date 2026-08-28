#!/usr/bin/env python3
"""
swin_umamba_adapter.py
=============================================================================
Experiment C Adapter: Swin-UMamba (2D Slice-Wise Strategy).
Uses 2D axial slice processing with 3-channel spatial context (slices i-1, i, i+1)
and 2D Swin / VMamba pretrained backbone, stacking 2D predictions into 3D volumes.
"""

import torch
import torch.nn as nn
import timm
from monai.networks.nets import ViTAutoEnc

class SwinUMamba2D(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super().__init__()
        # 2D backbone with ImageNet pretrained Swin/Vision Transformer encoder
        self.encoder = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, in_chans=in_channels, features_only=True)
        
        # 2D Decoder blocks
        self.conv1 = nn.Conv2d(768, 384, kernel_size=3, padding=1)
        self.up1 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(192, 96, kernel_size=3, padding=1)
        self.up2 = nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(48, 24, kernel_size=3, padding=1)
        self.up3 = nn.ConvTranspose2d(24, 24, kernel_size=8, stride=8) # Upsample back to input size
        
        self.out_head = nn.Conv2d(24, out_channels, kernel_size=1)

    def forward(self, x):
        # x shape: (B, 3, H, W) 2D slice context
        feats = self.encoder(x)
        bot = feats[-1] # Deepest feature map
        
        x = self.conv1(bot)
        x = self.up1(x)
        x = self.conv2(x)
        x = self.up2(x)
        x = self.conv3(x)
        x = self.up3(x)
        
        if x.shape[-2:] != x.shape[-2:]:
            x = nn.functional.interpolate(x, size=x.shape[-2:], mode="bilinear", align_corners=False)
            
        return self.out_head(x)

class SwinUMamba3DWrapper(nn.Module):
    """
    Wrapper that processes 3D volumes (B, C, Z, Y, X) slice-by-slice along axial Z-axis
    using 3-channel slice context, and outputs 3D logits (B, out_channels, Z, Y, X).
    """
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        self.net2d = SwinUMamba2D(in_channels=3, out_channels=out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        # x shape: (B, 1, Z, Y, X)
        b, c, z, y, x_dim = x.shape
        x_squeezed = x.squeeze(1) # (B, Z, Y, X)
        
        # Prepare 3-channel slice context [z-1, z, z+1]
        z_prev = torch.cat([x_squeezed[:, :1, :, :], x_squeezed[:, :-1, :, :]], dim=1)
        z_curr = x_squeezed
        z_next = torch.cat([x_squeezed[:, 1:, :, :], x_squeezed[:, -1:, :, :]], dim=1)
        
        # Stack to (B, Z, 3, Y, X)
        context = torch.stack([z_prev, z_curr, z_next], dim=2)
        
        # Flatten batch and depth: (B * Z, 3, Y, X)
        context_flat = context.view(b * z, 3, y, x_dim)
        
        # Resize to 224x224 for Swin input requirement if necessary
        if (y, x_dim) != (224, 224):
            context_flat = nn.functional.interpolate(context_flat, size=(224, 224), mode="bilinear", align_corners=False)
            
        logits_2d = self.net2d(context_flat) # (B * Z, out_channels, 224, 224)
        
        # Resize logits back to (Y, X)
        if (y, x_dim) != (224, 224):
            logits_2d = nn.functional.interpolate(logits_2d, size=(y, x_dim), mode="bilinear", align_corners=False)
            
        # Reshape to 3D output: (B, out_channels, Z, Y, X)
        logits_3d = logits_2d.view(b, z, self.out_channels, y, x_dim).permute(0, 2, 1, 3, 4)
        return logits_3d

def build_swin_umamba_model(in_channels=1, out_channels=2):
    return SwinUMamba3DWrapper(in_channels=in_channels, out_channels=out_channels)
