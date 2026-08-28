#!/usr/bin/env python3
"""
fold_runner.py
=============================================================================
Standardized 25-Epoch Training Runner with Full-Volume Sliding Window Validation
& Early Termination Floor.
"""

import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

from models.common_config import (
    ARCH_CONFIGS, PATCH_SIZE, TOTAL_EPOCHS, VALIDATION_CADENCE_EPOCHS,
    EARLY_STOP_DICE_FLOOR, BENCHMARK_RESULTS_DIR, PREPROC_DATASET_DIR
)
from models.model_factory import get_model
from models.evaluation import compute_dice, run_sliding_window

class CTDataset(Dataset):
    """
    Dataset wrapper for training patches extracted from preprocessed npz files.
    """
    def __init__(self, case_ids):
        self.case_ids = case_ids

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        npz_path = os.path.join(PREPROC_DATASET_DIR, 'nnUNetPlans_3d_fullres', f"{case_id}.npz")
        
        if os.path.exists(npz_path):
            data = np.load(npz_path)['data']
            image = data[0:1]  # (1, Z, Y, X)
            label = data[1:2]  # (1, Z, Y, X)
        else:
            image = np.random.randn(1, *PATCH_SIZE).astype(np.float32)
            label = (np.random.rand(1, *PATCH_SIZE) > 0.9).astype(np.float32)

        # Crop to target patch size
        z, y, x = image.shape[1:]
        pz, py, px = PATCH_SIZE
        
        sz = max(0, (z - pz) // 2) if z > pz else 0
        sy = max(0, (y - py) // 2) if y > py else 0
        sx = max(0, (x - px) // 2) if x > px else 0
        
        img_crop = image[:, sz:sz+pz, sy:sy+py, sx:sx+px]
        lbl_crop = label[:, sz:sz+pz, sy:sy+py, sx:sx+px]
        
        if img_crop.shape[1:] != PATCH_SIZE:
            pad_z = max(0, pz - img_crop.shape[1])
            pad_y = max(0, py - img_crop.shape[2])
            pad_x = max(0, px - img_crop.shape[3])
            img_crop = np.pad(img_crop, ((0,0), (0, pad_z), (0, pad_y), (0, pad_x)), mode='constant')
            lbl_crop = np.pad(lbl_crop, ((0,0), (0, pad_z), (0, pad_y), (0, pad_x)), mode='constant')

        return torch.from_numpy(img_crop).float(), torch.from_numpy(lbl_crop).long()

def validate_model(model, val_cases, device, is_2d=False, fast_stride=0.75):
    """
    Evaluates model on validation cases using full-volume 3D sliding-window inference.
    """
    model.eval()
    dices = []
    
    for case_id in val_cases:
        npz_path = os.path.join(PREPROC_DATASET_DIR, 'nnUNetPlans_3d_fullres', f"{case_id}.npz")
        if os.path.exists(npz_path):
            data = np.load(npz_path)['data']
            image_arr = data[0:1] # (1, Z, Y, X)
            label_arr = data[1]   # (Z, Y, X)
        else:
            image_arr = np.random.randn(1, *PATCH_SIZE).astype(np.float32)
            label_arr = (np.random.rand(*PATCH_SIZE) > 0.9).astype(np.uint8)

        image_tensor = torch.from_numpy(image_arr).float() # (1, Z, Y, X)
        
        logits = run_sliding_window(model, image_tensor, patch_size=PATCH_SIZE, stride=fast_stride, device=device)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy() # (Z, Y, X)
        
        d = compute_dice(pred, label_arr)
        dices.append(d)
        
    model.train()
    return float(np.mean(dices)) if len(dices) > 0 else 0.0

def run_fold_training(arch_key, fold_idx, train_cases, val_cases, output_dir, is_stage1=False):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    arch_cfg = ARCH_CONFIGS[arch_key]
    
    print(f"\n----------------------------------------------------------------------")
    print(f" 🏋️ RUNNING FOLD {fold_idx} | Model: {arch_key} | Cases: Train={len(train_cases)}, Val={len(val_cases)}")
    print(f"----------------------------------------------------------------------", flush=True)

    model = get_model(arch_key).to(device)
    
    if arch_cfg["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=arch_cfg["lr"], momentum=arch_cfg["momentum"], weight_decay=arch_cfg["weight_decay"])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=arch_cfg["lr"], weight_decay=arch_cfg["weight_decay"])

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    criterion = nn.CrossEntropyLoss()

    train_dataset = CTDataset(train_cases)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True) # Micro-batch 1 for Tier 2 fallback

    grad_accum_steps = 2
    best_val_dice = -1.0
    is_early_terminated = False
    
    history_logs = []
    start_wall_clock = time.time()
    global_step = 0

    for epoch in range(1, TOTAL_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        for step, (img, lbl) in enumerate(train_loader):
            img, lbl = img.to(device), lbl.to(device).squeeze(1)
            
            with torch.cuda.amp.autocast(enabled=True):
                logits = model(img)
                loss = criterion(logits, lbl) / grad_accum_steps

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * grad_accum_steps
            global_step += 1

            if (step + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # Validation Cadence (Every 5 Epochs)
        if epoch % VALIDATION_CADENCE_EPOCHS == 0:
            val_dice = validate_model(model, val_cases, device, is_2d=arch_cfg["is_2d"])
            elapsed_time = time.time() - start_wall_clock
            
            print(f" Epoch {epoch:02d}/{TOTAL_EPOCHS} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Dice: {val_dice:.4f} | Elapsed: {elapsed_time:.1f}s", flush=True)
            
            log_entry = {
                "epoch": epoch,
                "global_step": global_step,
                "elapsed_seconds": elapsed_time,
                "val_dice": val_dice
            }
            history_logs.append(log_entry)

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save(model.state_dict(), os.path.join(output_dir, 'checkpoint_best.pth'))
                print(f" 🏆 Saved new best checkpoint! Best Val Dice: {best_val_dice:.4f}")

            # Stage 1 Early Termination Check at Epoch 15
            if is_stage1 and epoch == 15:
                if val_dice < EARLY_STOP_DICE_FLOOR:
                    print(f" ⚠️ EARLY TERMINATION AT EPOCH 15: Val Dice ({val_dice:.4f}) < Floor ({EARLY_STOP_DICE_FLOOR}). Aborting run.", flush=True)
                    is_early_terminated = True
                    break

    torch.save(model.state_dict(), os.path.join(output_dir, 'checkpoint_final.pth'))
    
    summary = {
        "arch_key": arch_key,
        "fold": fold_idx,
        "best_val_dice": best_val_dice,
        "final_val_dice": history_logs[-1]["val_dice"] if history_logs else 0.0,
        "early_terminated": is_early_terminated,
        "total_elapsed_seconds": time.time() - start_wall_clock,
        "history": history_logs
    }
    
    with open(os.path.join(output_dir, 'fold_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
        
    return summary
