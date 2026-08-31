#!/usr/bin/env python3
"""
test_ab_isolation.py
=============================================================================
A/B Isolation Diagnostic Script:
Tests 4 combinations over 5 epochs to isolate whether Augmentation or Deep Supervision
is responsible for performance drop.
"""

import os, json, time
import torch
from models.common_config import STAGE1_FIXED_VAL_CASES, BENCHMARK_RESULTS_DIR, SPLITS_FINAL_PATH
from models.fold_runner import run_fold_training
from models.model_factory import get_model

splits = json.load(open(SPLITS_FINAL_PATH))
train_cases = splits[0]['train']
val_cases = STAGE1_FIXED_VAL_CASES

experiments = [
    {"name": "1_Baseline_NoAug_NoDS", "arch": "A_nnUNet", "augment": False},
    {"name": "2_AugOnly_NoDS",       "arch": "A_nnUNet", "augment": True},
]

print("======================================================================")
print(" 🔬 A/B ISOLATION DIAGNOSTIC BENCHMARK (5 EPOCHS)")
print("======================================================================\n")

for exp in experiments:
    print(f" ▶️ Testing {exp['name']}...")
    out_dir = os.path.join(BENCHMARK_RESULTS_DIR, 'ab_test', exp['name'])
    # run 5 epochs
