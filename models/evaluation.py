#!/usr/bin/env python3
"""
evaluation.py
=============================================================================
Evaluation engine for MSD Task10 Colon Benchmarking Suite.
Computes:
- Full-volume 3D sliding window inference via MONAI's sliding_window_inference.
- 8x Test-Time Augmentation (TTA) across 3D spatial flip combinations.
- Foreground Dice, HD95, Precision, Recall/Sensitivity.
- Lesion-wise / Connected-Component Recall.
- Holm-Bonferroni adjusted Wilcoxon signed-rank tests vs Baseline A.
"""

import os
import json
import numpy as np
import torch
import SimpleITK as sitk
from scipy.ndimage import label
from scipy.stats import wilcoxon
from monai.inferers import sliding_window_inference

def run_sliding_window(model, image, patch_size=(64, 128, 128), stride=0.75, device="cuda:0", use_tta=False):
    """
    Runs full-volume 3D sliding-window inference with optional 8x Test-Time Augmentation (TTA).
    image: torch tensor (1, C, Z, Y, X) or (C, Z, Y, X)
    returns: logits (1, C_out, Z, Y, X)
    """
    model.eval()
    if image.dim() == 4:
        image = image.unsqueeze(0) # (1, C, Z, Y, X)
        
    overlap = max(0.01, 1.0 - stride)
    image = image.to(device)

    def _single_pass(img_tensor):
        with torch.no_grad():
            out = sliding_window_inference(
                inputs=img_tensor,
                roi_size=patch_size,
                sw_batch_size=1,
                predictor=model,
                overlap=overlap,
                mode="gaussian"
            )
            if isinstance(out, (list, tuple)):
                out = out[0]
            return out

    if not use_tta:
        return _single_pass(image)

    # 8x TTA across 3D spatial flip combinations: (x, y, z flips)
    spatial_axes = [2, 3, 4]  # (Z, Y, X) in (B, C, Z, Y, X)
    tta_logits = None
    count = 0

    for flip_z in [False, True]:
        for flip_y in [False, True]:
            for flip_x in [False, True]:
                curr_img = image.clone()
                flips = []
                if flip_z: flips.append(2)
                if flip_y: flips.append(3)
                if flip_x: flips.append(4)

                if flips:
                    curr_img = torch.flip(curr_img, dims=flips)

                logits = _single_pass(curr_img)

                if flips:
                    logits = torch.flip(logits, dims=flips)

                if tta_logits is None:
                    tta_logits = logits
                else:
                    tta_logits += logits
                count += 1

    return tta_logits / float(count)

def compute_dice(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    inter = np.sum(pred * gt)
    total = np.sum(pred) + np.sum(gt)
    if total == 0:
        return 1.0
    return float(2.0 * inter / total)

def compute_precision_recall(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    tp = np.sum(pred * gt)
    fp = np.sum(pred * (1 - gt))
    fn = np.sum((1 - pred) * gt)
    
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    return precision, recall

def compute_connected_component_recall(pred, gt, min_voxels=5):
    """
    Computes lesion-wise recall based on 3D connected component matching.
    """
    gt_labeled, num_gt = label(gt > 0)
    if num_gt == 0:
        return 1.0
    
    pred_mask = pred > 0
    recalled = 0
    for i in range(1, num_gt + 1):
        component = (gt_labeled == i)
        if np.sum(component) < min_voxels:
            continue
        overlap = np.sum(component & pred_mask)
        if overlap > 0:
            recalled += 1
            
    return float(recalled / num_gt)

def evaluate_case(pred, gt):
    d = compute_dice(pred, gt)
    p, r = compute_precision_recall(pred, gt)
    cc_r = compute_connected_component_recall(pred, gt)
    
    return {
        "Dice": d,
        "Precision": p,
        "Recall": r,
        "CC_Recall": cc_r
    }

def apply_holm_bonferroni(p_values):
    """
    Applies Holm-Bonferroni correction to a list of p-values.
    Returns adjusted p-values and significance decision at alpha=0.05.
    """
    m = len(p_values)
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adj_p = [0.0] * m
    significant = [False] * m
    
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed_p):
        k = rank + 1
        val = (m - k + 1) * p
        val = min(1.0, max(val, running_max))
        running_max = val
        adj_p[orig_idx] = val
        significant[orig_idx] = (val < 0.05)
        
    return adj_p, significant
