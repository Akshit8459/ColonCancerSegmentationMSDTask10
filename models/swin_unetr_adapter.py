#!/usr/bin/env python3
"""
swin_unetr_adapter.py
=============================================================================
Experiment E Adapter: MONAI Swin-UNETR 3D Transformer with SSL Pretrained Weights.
"""

import torch
from monai.networks.nets import SwinUNETR

def build_swin_unetr_model(in_channels=1, out_channels=2):
    model = SwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=48,
        use_checkpoint=True,
        spatial_dims=3
    )
    
    # Load MONAI SSL pretrained weights
    url = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt"
    try:
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu', progress=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        print(" ✅ Loaded MONAI SSL 3D pretrained weights for SwinUNETR.")
    except Exception as e:
        print(f" ⚠️ Could not load SwinUNETR SSL pretrained weights: {e}")
        
    return model