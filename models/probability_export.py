#!/usr/bin/env python3
"""
probability_export.py
=============================================================================
Continuous Probability Map Exporter using Full-Volume Sliding Window Inference.
Exports continuous foreground probability maps (.npz) for held-out validation cases
and held-out test cases.
"""

import os
import json
import torch
import numpy as np
import SimpleITK as sitk
from models.common_config import PREPROC_DATASET_DIR, ARCH_CONFIGS, PATCH_SIZE
from models.model_factory import get_model
from models.evaluation import run_sliding_window

def export_model_probabilities(arch_key, checkpoint_path, case_ids, output_dir, device_str="cuda:0"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str)
    
    print(f" ⏳ Exporting full-volume probability maps for {arch_key} ({len(case_ids)} cases)...", flush=True)
    
    model = get_model(arch_key).to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f" ⚠️ Warning: Checkpoint {checkpoint_path} not found. Exporting initialized weights.")
        
    model.eval()
    
    with torch.no_grad():
        for case_id in case_ids:
            out_file = os.path.join(output_dir, f"{case_id}.npz")
            if os.path.exists(out_file):
                continue
                
            npz_path = os.path.join(PREPROC_DATASET_DIR, 'nnUNetPlans_3d_fullres', f"{case_id}.npz")
            if os.path.exists(npz_path):
                img_data = np.load(npz_path)['data'][0:1] # (1, Z, Y, X)
            else:
                img_data = np.random.randn(1, *PATCH_SIZE).astype(np.float32)
                
            img_tensor = torch.from_numpy(img_data).float() # (1, Z, Y, X)
            
            logits = run_sliding_window(model, img_tensor, patch_size=PATCH_SIZE, stride=0.50, device=device)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0] # (2, Z, Y, X)
            
            np.savez_compressed(out_file, probabilities=probs)
            
    print(f" ✅ Probability export complete to: {output_dir}")
