#!/usr/bin/env python3
"""
swin_unetr_adapter.py
=============================================================================
Experiment E Adapter: MONAI Swin-UNETR 3D Transformer.
Loads MONAI 3D Swin-UNETR architecture.
"""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

def build_swin_unetr_model(in_channels=1, out_channels=2):
    model = SwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=48,
        use_checkpoint=True,
        spatial_dims=3
    )
    return model
