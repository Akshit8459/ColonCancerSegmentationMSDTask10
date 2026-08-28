#!/usr/bin/env python3
"""
nnunet_adapter.py
=============================================================================
Experiment A Adapter: Standard 3D nnU-Net Baseline.
Uses 3D UNet with SGD + Nesterov momentum + Polynomial LR schedule.
"""

import torch
import torch.nn as nn
from monai.networks.nets import UNet

class nnUNetAdapter(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        # Standard 3D UNet matching nnU-Net 3d_fullres depth
        self.net = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(32, 64, 128, 256, 320, 320),
            strides=(2, 2, 2, 2, 2),
            num_res_units=2,
            norm="instance"
        )

    def forward(self, x):
        return self.net(x)

def build_nnunet_model(in_channels=1, out_channels=2):
    return nnUNetAdapter(in_channels=in_channels, out_channels=out_channels)
