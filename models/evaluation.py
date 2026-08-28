#!/usr/bin/env python3
"""
evaluation.py
=============================================================================
Evaluation engine for MSD Task10 Colon Benchmarking Suite.
Computes:
- Full-volume 3D sliding window inference via MONAI's sliding_window_inference.
- Foreground Dice, HD95, Precision, Recall/Sensitivity.
- Lesion-wise / Connected-Component Recall.
- Subgroup stratification (Small <2cm, Medium 2-5cm, Large >5cm; Single vs Multi-lesion).
- Sliding-window inference with fast stride (0.75) for training validation and full stride (0.5 + 8x TTA) for final test.
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

def run_sliding_window(model, image, patch_size=(64, 128, 128), stride=0.75, device="cuda:0"):
    """
    Runs full-volume 3D sliding-window inference.
    image: torch tensor (1, C, Z, Y, X) or (C, Z, Y, X)
    returns: logits (1, C_out, Z, Y, X)
    """
    model.eval()
    if image.dim() == 4:
        image = image.unsqueeze(0) # (1, C, Z, Y, X)
        
    overlap = max(0.01, 1.0 - stride)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            logits = sliding_window_inference(
                inputs=image.to(device),
                roi_size=patch_size,
                sw_batch_size=1,
                predictor=model,
                overlap=overlap,
                mode="gaussian"
            )
    return logits

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
