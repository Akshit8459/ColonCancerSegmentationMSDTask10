#!/usr/bin/env python3
"""
model_factory.py
=============================================================================
Factory function to build model instances for Experiments A–F.
"""

from models.nnunet_adapter import build_nnunet_model
from models.umamba_adapter import build_umamba_model
from models.swin_umamba_adapter import build_swin_umamba_model
from models.segmamba_adapter import build_segmamba_model
from models.swin_unetr_adapter import build_swin_unetr_model
from models.nnuzoo_adapter import build_nnuzoo_model

def get_model(arch_key, in_channels=1, out_channels=2):
    if arch_key == "A_nnUNet":
        return build_nnunet_model(in_channels, out_channels)
    elif arch_key == "B_UMamba":
        return build_umamba_model(in_channels, out_channels)
    elif arch_key == "C_SwinUMamba":
        return build_swin_umamba_model(in_channels, out_channels)
    elif arch_key == "D_SegMamba":
        return build_segmamba_model(in_channels, out_channels)
    elif arch_key == "E_SwinUNETR":
        return build_swin_unetr_model(in_channels, out_channels)
    elif arch_key == "F_nnUZoo":
        return build_nnuzoo_model(in_channels, out_channels)
    else:
        raise ValueError(f"Unknown architecture key: {arch_key}")
