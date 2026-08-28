#!/usr/bin/env python3
"""
probability_export.py
=============================================================================
Continuous Probability Map Exporter using Full-Volume Sliding Window Inference
with optional 8x Test-Time Augmentation (TTA).
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from .common_config import PREPROC_DATASET_DIR, ARCH_CONFIGS, PATCH_SIZE
from .model_factory import get_model
from .evaluation import run_sliding_window
from .fold_runner import load_case_data

def export_model_probabilities(arch_key, checkpoint_path, case_ids, output_dir, device_str="cuda:0", use_tta=False):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str)
    
    model = get_model(arch_key).to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f" ⚠️ Warning: Checkpoint {checkpoint_path} not found. Exporting initialized weights.")
        
    model.eval()
    
    with torch.no_grad():
        pbar = tqdm(case_ids, desc=f" ⏳ Exporting Probs [{arch_key}]", leave=True)
        for case_id in pbar:
            out_file = os.path.join(output_dir, f"{case_id}.npz")
            if os.path.exists(out_file):
                continue
                
            image_arr, _ = load_case_data(case_id) # (1, Z, Y, X)
            img_tensor = torch.from_numpy(image_arr).float()
            
            logits = run_sliding_window(model, img_tensor, patch_size=PATCH_SIZE, stride=0.50, device=device, use_tta=use_tta)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0] # (2, Z, Y, X)
            
            np.savez_compressed(out_file, probabilities=probs)
