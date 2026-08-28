#!/usr/bin/env python3
"""
debug_validation.py
=============================================================================
Validation & Patch Sampling Debugging Diagnostic Script.
"""

import os
import sys
import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from models.fold_runner import load_case_data, validate_model, CTDataset
from models.model_factory import get_model
from models.common_config import PATCH_SIZE, ARCH_CONFIGS, STAGE1_FIXED_VAL_CASES
from models.evaluation import run_sliding_window, compute_dice

def main():
    print("======================================================================")
    print(" 🛠️ RUNNING VALIDATION & DATASET DEBUG DIAGNOSTIC")
    print("======================================================================")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f" Target Device: {device}\n")

    # 1. Inspect Ground Truth Loading
    case_id = STAGE1_FIXED_VAL_CASES[0]
    image_arr, label_arr = load_case_data(case_id)
    
    print(f" 📦 Case ID: {case_id}")
    print(f"    - Image Shape: {image_arr.shape}, Dtype: {image_arr.dtype}, Min: {image_arr.min():.2f}, Max: {image_arr.max():.2f}")
    print(f"    - Label Shape: {label_arr.shape}, Dtype: {label_arr.dtype}, Unique: {np.unique(label_arr)}, Sum: {label_arr.sum()}")

    fg_voxels = (label_arr == 1).sum()
    print(f"    - Foreground Voxels (class 1): {fg_voxels}")
    
    # 2. Inspect 50% Foreground Patch Sampling
    dataset = CTDataset([case_id])
    fg_patches = 0
    total_patches = 20
    for _ in range(total_patches):
        img_patch, lbl_patch = dataset[0]
        if (lbl_patch == 1).sum() > 0:
            fg_patches += 1
    print(f"\n 🎯 50% Foreground Sampling Check ({total_patches} samples):")
    print(f"    - Patches containing foreground: {fg_patches}/{total_patches} ({fg_patches/total_patches*100:.1f}%)\n")

    # 3. Test Model Inference & Sliding Window
    print(" 🚀 Initializing A_nnUNet model for sliding window inference...")
    model = get_model("A_nnUNet").to(device)
    model.eval()

    image_tensor = torch.from_numpy(image_arr).float().to(device)
    
    with torch.no_grad():
        print(" ⏳ Running sliding window inference...")
        logits = run_sliding_window(model, image_tensor, patch_size=PATCH_SIZE, stride=0.75, device=device)
        print(f"    - Logits Shape: {logits.shape}")
        print(f"    - Logits Min: {logits.min().item():.4f}, Max: {logits.max().item():.4f}")
        
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        print(f"    - Prediction Unique Values: {np.unique(pred)}")
        print(f"    - Prediction Foreground Voxels: {(pred == 1).sum()}")

        lbl = label_arr[0]
        dice = compute_dice(pred, lbl)
        print(f"\n 📊 Initialized Model Validation Dice: {dice:.4f}")
        print("======================================================================\n")

if __name__ == '__main__':
    main()
