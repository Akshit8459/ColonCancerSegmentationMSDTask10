#!/usr/bin/env python3
"""
fast_diagnostic_test.py
=============================================================================
Ultra-Fast 30-Second Diagnostic Test to isolate Augmentation vs Deep Supervision.
Runs 1 epoch with a small 2-patches/volume budget (~20 steps = 25 seconds per test).
"""

import os, sys, time, json
import torch

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from models.common_config import STAGE1_FIXED_VAL_CASES, SPLITS_FINAL_PATH, ARCH_CONFIGS, MODEL_MAX_BATCH
from models.fold_runner import CTDataset, validate_model, load_case_data
from models.model_factory import get_model
from torch.utils.data import DataLoader, RandomSampler
from monai.losses import DiceCELoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
splits = json.load(open(SPLITS_FINAL_PATH))
train_cases = splits[0]['train'][:20] # Mini training subset of 20 volumes for fast test
val_cases = STAGE1_FIXED_VAL_CASES

print(" 📦 Caching 20 mini-training volumes...", flush=True)
volume_cache = {c: load_case_data(c) for c in train_cases}

conditions = [
    {"name": "Condition 1 (Baseline: No Aug, No DS)", "arch": "A_nnUNet", "augment": False},
    {"name": "Condition 2 (Augment Only: Aug, No DS)", "arch": "A_nnUNet", "augment": True},
    {"name": "Condition 3 (Deep Sup Only: No Aug, DS)", "arch": "G_nnUNet", "augment": False},
    {"name": "Condition 4 (Both: Aug, DS)",             "arch": "G_nnUNet", "augment": True},
]

print("\n======================================================================")
print(" ⚡ ULTRA-FAST 30-SECOND DIAGNOSTIC TEST (1 Mini-Epoch / 20 Steps)")
print("======================================================================\n", flush=True)

results = {}

for cond in conditions:
    print(f" ▶️ Running {cond['name']}...", flush=True)
    t0 = time.time()
    
    dataset = CTDataset(train_cases, volume_cache=volume_cache, augment=cond['augment'])
    sampler = RandomSampler(dataset, replacement=True, num_samples=len(train_cases) * 2) # 40 patches total
    loader = DataLoader(dataset, batch_size=4, sampler=sampler, drop_last=True)
    
    model = get_model(cond['arch']).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99)
    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    
    model.train()
    for step, (img, lbl) in enumerate(loader):
        img, lbl = img.to(device), lbl.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            logits = model(img)
            if isinstance(logits, torch.Tensor) and logits.dim() == 6:
                logits = [logits[:, :, i] for i in range(logits.shape[2])]
            
            if isinstance(logits, (list, tuple)):
                losses = []
                for out in logits:
                    scale = out.shape[2:]
                    lbl_scaled = torch.nn.functional.interpolate(lbl.float(), size=scale, mode='nearest').long()
                    losses.append(criterion(out, lbl_scaled))
                loss = sum(losses) / float(len(logits))
            else:
                loss = criterion(logits, lbl)
        loss.backward()
        optimizer.step()
    
    val_dice = validate_model(model, val_cases, device, is_2d=False)
    elapsed = time.time() - t0
    results[cond['name']] = val_dice
    print(f"   --> Val Dice: {val_dice:.4f} | Done in {elapsed:.1f}s\n", flush=True)

print("======================================================================")
print(" 📊 DIAGNOSTIC RESULTS SUMMARY (1 Mini-Epoch Val Dice)")
print("======================================================================")
for name, d in results.items():
    print(f"  • {name:42s}: Val Dice = {d:.4f}")
print("======================================================================\n")
