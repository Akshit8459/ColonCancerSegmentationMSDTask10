#!/usr/bin/env python3
"""
nnunet_adapter.py
=============================================================================
Experiment A Adapter: Official nnU-Net 3D Architecture via MONAI DynUNet.
Matches nnU-Net 3d_fullres kernel sizes, strides, InstanceNorm, and LeakyReLU.
"""

import torch.nn as nn
from monai.networks.nets import DynUNet

class nnUNetAdapter(nn.Module):
    """
    nnU-Net 3D full-resolution architecture proxy via MONAI DynUNet.
    """
    def __init__(self, in_channels=1, out_channels=2, deep_supervision=False):
        super().__init__()
        self.net = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[[3,3,3], [3,3,3], [3,3,3], [3,3,3], [3,3,3]],
            strides=[[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2]],
            upsample_kernel_size=[[2,2,2], [2,2,2], [2,2,2], [2,2,2]],
            norm_name='instance',
            act_name='leakyrelu',
            deep_supervision=deep_supervision
        )

    def forward(self, x):
        return self.net(x)

def build_nnunet_model(in_channels=1, out_channels=2, deep_supervision=False):
    return nnUNetAdapter(in_channels=in_channels, out_channels=out_channels, deep_supervision=deep_supervision)
