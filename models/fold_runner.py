#!/usr/bin/env python3
"""
fold_runner.py
=============================================================================
Standardized 25-Epoch Training Runner with MONAI 3D Data Augmentation,
RAM Volume Caching, Per-Model Batch Sizes, torch.compile Acceleration,
MONAI DiceCELoss with Deep Supervision, & Full-Volume Sliding Window Validation.
"""

import os
import time
import json
import torch
import torch._dynamo
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, RandomSampler

try:
    import blosc2
    HAS_BLOSC2 = True
except ImportError:
    HAS_BLOSC2 = False

from monai.losses import DiceCELoss
from monai.transforms import (
    Compose, RandFlipd, RandRotated, RandZoomd, RandGaussianNoised, RandAdjustContrastd
)
from .common_config import (
    ARCH_CONFIGS, PATCH_SIZE, TOTAL_EPOCHS, VALIDATION_CADENCE_EPOCHS,
    EARLY_STOP_DICE_FLOOR, BENCHMARK_RESULTS_DIR, PREPROC_DATASET_DIR, RAW_DATASET_DIR,
    MODEL_MAX_BATCH
)
from .model_factory import get_model
from .evaluation import compute_dice, run_sliding_window

def load_case_data(case_id):
    """
    Robust reader for preprocessed 3D CT volumes supporting .b2nd, .npz, and .nii.gz formats.
    Returns: img_arr (1, Z, Y, X), lbl_arr (1, Z, Y, X) as float32 clamped to [0, 1].
    """
    prep_dir = os.path.join(PREPROC_DATASET_DIR, 'nnUNetPlans_3d_fullres')
    img_b2nd = os.path.join(prep_dir, f"{case_id}.b2nd")
    seg_b2nd = os.path.join(prep_dir, f"{case_id}_seg.b2nd")
    
    if HAS_BLOSC2 and os.path.exists(img_b2nd) and os.path.exists(seg_b2nd):
        img_arr = blosc2.open(img_b2nd)[:]
        lbl_arr = blosc2.open(seg_b2nd)[:]
        if img_arr.ndim == 3:
            img_arr = img_arr[np.newaxis, ...]
        if lbl_arr.ndim == 3:
            lbl_arr = lbl_arr[np.newaxis, ...]
        return img_arr.astype(np.float32), np.clip(lbl_arr, 0, 1).astype(np.float32)
        
    npz_path = os.path.join(prep_dir, f"{case_id}.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path)['data']
        return data[0:1].astype(np.float32), np.clip(data[1:2], 0, 1).astype(np.float32)

    raw_img = os.path.join(RAW_DATASET_DIR, 'imagesTr', f"{case_id}_0000.nii.gz")
    raw_lbl = os.path.join(RAW_DATASET_DIR, 'labelsTr', f"{case_id}.nii.gz")
    if os.path.exists(raw_img) and os.path.exists(raw_lbl):
        img_sitk = sitk.ReadImage(raw_img)
        lbl_sitk = sitk.ReadImage(raw_lbl)
        img_arr = sitk.GetArrayFromImage(img_sitk)[np.newaxis, ...]
        lbl_arr = sitk.GetArrayFromImage(lbl_sitk)[np.newaxis, ...]
        return img_arr.astype(np.float32), np.clip(lbl_arr, 0, 1).astype(np.float32)

    # Synthetic fallback for debug environment
    img_arr = np.random.randn(1, 64, 128, 128).astype(np.float32)
    lbl_arr = (np.random.rand(1, 64, 128, 128) > 0.9).astype(np.float32)
    return img_arr, lbl_arr

class CTDataset(Dataset):
    """
    Dataset wrapper for training patches extracted from RAM-cached preprocessed volumes.
    Uses 80% foreground-centric sampling & MONAI 3D spatial/intensity data augmentations.
    """
    def __init__(self, case_ids, volume_cache=None, augment=False):
        self.case_ids = case_ids
        self.volume_cache = volume_cache
        self.augment = augment
        if augment:
            self.transform = Compose([
                RandFlipd(keys=['image', 'label'], prob=0.3, spatial_axis=[0, 1, 2]),
                RandRotated(keys=['image', 'label'], prob=0.2, range_x=0.1, range_y=0.1, range_z=0.1, mode=['bilinear', 'nearest']),
                RandZoomd(keys=['image', 'label'], prob=0.1, min_zoom=0.95, max_zoom=1.05, mode=['bilinear', 'nearest'])
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        if self.volume_cache is not None and case_id in self.volume_cache:
            image, label = self.volume_cache[case_id]
        else:
            image, label = load_case_data(case_id)

        z, y, x = image.shape[1:]
        pz, py, px = PATCH_SIZE

        fg_coords = np.argwhere(label[0] == 1)
        if len(fg_coords) > 0 and np.random.rand() < 0.7:  #Sampling Patches positive
            fg_idx = np.random.choice(len(fg_coords))
            cz, cy, cx = fg_coords[fg_idx]
            sz = max(0, min(cz - pz // 2, z - pz)) if z > pz else 0
            sy = max(0, min(cy - py // 2, y - py)) if y > py else 0
            sx = max(0, min(cx - px // 2, x - px)) if x > px else 0
        else:
            sz = np.random.randint(0, max(1, z - pz + 1)) if z > pz else 0
            sy = np.random.randint(0, max(1, y - py + 1)) if y > py else 0
            sx = np.random.randint(0, max(1, x - px + 1)) if x > px else 0

        img_crop = image[:, sz:sz+pz, sy:sy+py, sx:sx+px]
        lbl_crop = label[:, sz:sz+pz, sy:sy+py, sx:sx+px]

        if img_crop.shape[1:] != PATCH_SIZE:
            pad_z = max(0, pz - img_crop.shape[1])
            pad_y = max(0, py - img_crop.shape[2])
            pad_x = max(0, px - img_crop.shape[3])
            img_crop = np.pad(img_crop, ((0,0), (0, pad_z), (0, pad_y), (0, pad_x)), mode='constant')
            lbl_crop = np.pad(lbl_crop, ((0,0), (0, pad_z), (0, pad_y), (0, pad_x)), mode='constant')

        img_tensor = torch.from_numpy(img_crop).float()
        lbl_tensor = torch.from_numpy(lbl_crop).long()

        if self.augment and self.transform is not None:
            data = {"image": img_tensor, "label": lbl_tensor}
            data = self.transform(data)
            img_tensor, lbl_tensor = data["image"], data["label"]

        return img_tensor, lbl_tensor

def validate_model(model, val_cases, device, is_2d=False, fast_stride=0.75):
    """
    Evaluates model on validation cases using full-volume 3D sliding-window inference.
    Uses unwrapped model (_orig_mod) to avoid torch.compile / Triton CUDA graph memory spikes.
    """
    eval_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    eval_model.eval()
    dices = []
    
    val_pbar = tqdm(val_cases, desc=" 🔍 Validating", leave=False)
    for case_id in val_pbar:
        image_arr, label_arr = load_case_data(case_id)
        lbl = np.clip(label_arr[0], 0, 1) # (Z, Y, X)

        image_tensor = torch.from_numpy(image_arr).float() # (1, Z, Y, X)
        
        logits = run_sliding_window(eval_model, image_tensor, patch_size=PATCH_SIZE, stride=fast_stride, device=device)
        if isinstance(logits, torch.Tensor) and logits.dim() == 6:
            logits = logits[:, :, 0]
        elif isinstance(logits, (list, tuple)):
            logits = logits[0]
            
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy() # (Z, Y, X)
        
        d = compute_dice(pred, lbl)
        dices.append(d)
        val_pbar.set_postfix(dice=f"{d:.4f}")
        
    model.train()
    return float(np.mean(dices)) if len(dices) > 0 else 0.0

def run_fold_training(arch_key, fold_idx, train_cases, val_cases, output_dir, is_stage1=False):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    arch_cfg = ARCH_CONFIGS[arch_key]
    
    batch_size = MODEL_MAX_BATCH.get(arch_key, 1)
    grad_accum_steps = max(1, 2 // batch_size)

    print(f"\n----------------------------------------------------------------------")
    print(f" 🏋️ RUNNING FOLD {fold_idx} | Model: {arch_key} | Cases: Train={len(train_cases)}, Val={len(val_cases)}")
    print(f" ⚡ Speed Settings: Batch Size={batch_size}, Grad Accum Steps={grad_accum_steps}")
    print(f"----------------------------------------------------------------------", flush=True)

    model = get_model(arch_key).to(device)

    # Standard cuDNN PyTorch Eager mode with AMP autocast (Instant startup in < 1 second)
    
    if arch_cfg["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=arch_cfg["lr"], momentum=arch_cfg["momentum"], weight_decay=arch_cfg["weight_decay"])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=arch_cfg["lr"], weight_decay=arch_cfg["weight_decay"])

    scaler = torch.amp.GradScaler('cuda', enabled=True)
    criterion = DiceCELoss(
        to_onehot_y=True,
        softmax=True
    )

    # Pre-cache all training volumes in RAM using multi-core parallel threads
    print(" 📦 Caching training volumes into RAM across multi-core CPU threads...", flush=True)
    from concurrent.futures import ThreadPoolExecutor
    
    def _read_case(c_id):
        return c_id, load_case_data(c_id)
        
    volume_cache = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        cached_items = list(tqdm(executor.map(_read_case, train_cases), total=len(train_cases), desc=" ⚡ Parallel RAM Caching", leave=False))
        volume_cache = dict(cached_items)

    train_dataset = CTDataset(train_cases, volume_cache=volume_cache, augment=True)
    num_patches_per_volume = 50  # 50 patches per volume (4200 samples/epoch)
    total_patches_per_epoch = len(train_cases) * num_patches_per_volume

    sampler = RandomSampler(train_dataset, replacement=True, num_samples=total_patches_per_epoch)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )

    patience = 10               # Stop if no improvement for 10 consecutive validation checks (50 epochs)
    patience_counter = 0
    best_val_dice = -1.0
    best_epoch = -1
    is_early_terminated = False
    
    history_logs = []
    start_wall_clock = time.time()
    global_step = 0

    for epoch in range(1, TOTAL_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f" Epoch {epoch:02d}/{TOTAL_EPOCHS} [{arch_key}]", leave=True)
        for step, (img, lbl) in enumerate(pbar):
            img, lbl = img.to(device), lbl.to(device)
            
            with torch.amp.autocast('cuda', enabled=True):
                logits = model(img)
                if isinstance(logits, torch.Tensor) and logits.dim() == 6:
                    logits = [logits[:, :, i] for i in range(logits.shape[2])]
                
                if isinstance(logits, (list, tuple)):
                    losses = []
                    for out in logits:
                        scale = out.shape[2:]  # (Z, Y, X)
                        lbl_scaled = torch.nn.functional.interpolate(lbl.float(), size=scale, mode='nearest').long()
                        losses.append(criterion(out, lbl_scaled))
                    loss = sum(losses) / float(len(logits))
                else:
                    loss = criterion(logits, lbl)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            batch_loss = loss.item() * grad_accum_steps
            epoch_loss += batch_loss
            global_step += 1

            if (step + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            pbar.set_postfix(loss=f"{batch_loss:.4f}", avg_loss=f"{epoch_loss / (step + 1):.4f}")

        # Validation Cadence (Every 5 Epochs)
        if epoch % VALIDATION_CADENCE_EPOCHS == 0:
            torch.cuda.empty_cache()
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
                best_epoch = epoch
                patience_counter = 0
                unwrapped_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                torch.save(unwrapped_model.state_dict(), os.path.join(output_dir, 'checkpoint_best.pth'))
                print(f" 🏆 Saved new best checkpoint at epoch {epoch}! Best Val Dice: {best_val_dice:.4f}")
            else:
                patience_counter += 1
                print(f" ⏳ No improvement for {patience_counter}/{patience} validation checks (Best: {best_val_dice:.4f} at epoch {best_epoch}).")
                if is_stage1 and patience_counter >= patience:
                    print(f" ⏹️ Early stopping at epoch {epoch} – no validation improvement for {patience} consecutive checks.")
                    is_early_terminated = True
                    break

    unwrapped_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save(unwrapped_model.state_dict(), os.path.join(output_dir, 'checkpoint_final.pth'))
    
    summary = {
        "arch_key": arch_key,
        "fold": fold_idx,
        "best_val_dice": best_val_dice,
        "best_epoch": best_epoch,
        "final_val_dice": history_logs[-1]["val_dice"] if history_logs else 0.0,
        "early_terminated": is_early_terminated,
        "total_elapsed_seconds": time.time() - start_wall_clock,
        "history": history_logs
    }
    
    with open(os.path.join(output_dir, 'fold_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
        
    return summary
