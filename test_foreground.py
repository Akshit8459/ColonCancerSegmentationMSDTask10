#!/usr/bin/env python3
"""
test_foreground.py
=============================================================================
Minimal test script to verify model foreground prediction dynamics.
"""

import os
import torch
import numpy as np
from models.fold_runner import load_case_data
from models.model_factory import get_model
from models.common_config import PATCH_SIZE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
arch_key = "A_nnUNet"

print(f"=== TESTING FOREGROUND DYNAMICS FOR {arch_key} ===")

# Load one training case
case_id = "colon_039"
image_arr, label_arr = load_case_data(case_id)
print(f"Full Volume Label unique values: {np.unique(label_arr)}")
print(f"Full Volume Label sum (foreground voxels): {label_arr.sum()}")

# Find a patch containing foreground voxels
fg_coords = np.argwhere(label_arr[0] == 1)
if len(fg_coords) > 0:
    cz, cy, cx = fg_coords[0]
    z, y, x = image_arr.shape[1:]
    pz, py, px = PATCH_SIZE
    sz = max(0, min(cz - pz // 2, z - pz)) if z > pz else 0
    sy = max(0, min(cy - py // 2, y - py)) if y > py else 0
    sx = max(0, min(cx - px // 2, x - px)) if x > px else 0
else:
    sz, sy, sx = 0, 0, 0

img_patch = image_arr[:, sz:sz+64, sy:sy+128, sx:sx+128]
lbl_patch = label_arr[:, sz:sz+64, sy:sy+128, sx:sx+128]
print(f"Patch Label sum (foreground voxels): {lbl_patch.sum()}")

# Build model
model = get_model(arch_key).to(device)
model.train()

img_t = torch.from_numpy(img_patch).unsqueeze(0).float().to(device)
lbl_t = torch.from_numpy(lbl_patch).unsqueeze(0).long().to(device)

with torch.amp.autocast('cuda', enabled=True):
    logits = model(img_t)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
    print(f"Initial Prediction unique values: {np.unique(pred)}")
    print(f"Initial Prediction sum (foreground voxels): {pred.sum()}")
