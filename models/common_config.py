#!/usr/bin/env python3
"""
common_config.py
=============================================================================
Central configuration module for MSD Task10 Colon Benchmarking Suite.
Defines paths, split configurations, patch size, fixed Stage 1 5-case validation subset,
VRAM fallback tiers, and architecture-tailored optimizer policies.
Target GPU: GPU 5 (via CUDA_VISIBLE_DEVICES=5).
"""

import os
import json

# Allow command-line CUDA_VISIBLE_DEVICES override (default to GPU 5 if unspecified)
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# Base paths for nnU-Net dataset structures
NNUNET_RAW = os.environ.get('nnUNet_raw', '/home/akshitp/preprocessing/nnUNet_raw')
NNUNET_PREPROCESSED = os.environ.get('nnUNet_preprocessed', '/home/akshitp/preprocessing/nnUNet_preprocessed')
NNUNET_RESULTS = os.environ.get('nnUNet_results', '/home/akshitp/preprocessing/nnUNet_results')

DATASET_NAME = "Dataset012_ColonBowelROI_HighRes2mm"
RAW_DATASET_DIR = os.path.join(NNUNET_RAW, DATASET_NAME)
PREPROC_DATASET_DIR = os.path.join(NNUNET_PREPROCESSED, DATASET_NAME)

BENCHMARK_RESULTS_DIR = '/home/akshitp/Benchmarking/results'
os.makedirs(BENCHMARK_RESULTS_DIR, exist_ok=True)

# Splits files
SPLITS_FINAL_PATH = os.path.join(PREPROC_DATASET_DIR, 'splits_final.json')
HELD_OUT_TEST_PATH = os.path.join(PREPROC_DATASET_DIR, 'held_out_stratified_test_cases.json')

# Input geometry and resolution
TARGET_SPACING = (1.0, 1.0, 2.0)  # mm (x, y, z)
PATCH_SIZE = (64, 128, 128)      # (z, y, x)

# Optimization and training budget parameters
MICRO_BATCH_SIZE = 2
FORCE_FALLBACK_TIER = 2          # Tier 2: Micro-batch 1 + 2x Grad Accumulation
STAGE2_MAX_FINALISTS = 2         # Exp A (Anchor) + Stage 1 Winner
EARLY_STOP_DICE_FLOOR = 0.45     # Epoch 15 early termination threshold in Stage 1

TOTAL_EPOCHS = 25
VALIDATION_CADENCE_EPOCHS = 5

# Safe maximum per-architecture batch sizes (measured on 32GB VRAM)
MODEL_MAX_BATCH = {
    "A_nnUNet": 8,
    "B_UMamba": 4,
    "C_SwinUMamba": 4,
    "D_SegMamba": 8,
    "E_SwinUNETR": 4,
    "F_nnUZoo": 4,
}

# Fixed performance-stratified 5-case validation subset from Fold 0 validation split
STAGE1_FIXED_VAL_CASES = ["colon_039", "colon_102", "colon_166", "colon_157", "colon_006"]

# Per-architecture optimizer and learning rate policies
ARCH_CONFIGS = {
    "A_nnUNet": {
        "name": "nnU-Net 3D Baseline",
        "optimizer": "SGD",
        "momentum": 0.99,
        "weight_decay": 3e-5,
        "lr": 0.01,
        "lr_scheduler": "poly",
        "poly_exponent": 0.9,
        "is_2d": False
    },
    "B_UMamba": {
        "name": "U-Mamba 3D",
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "lr": 1e-4,
        "lr_scheduler": "cosine",
        "warmup_epochs": 2,
        "is_2d": False
    },
    "C_SwinUMamba": {
        "name": "Swin-UMamba 2D Slice-Wise",
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "lr": 5e-5,
        "lr_scheduler": "cosine",
        "warmup_epochs": 2,
        "is_2d": True
    },
    "D_SegMamba": {
        "name": "SegMamba 3D",
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "lr": 1e-4,
        "lr_scheduler": "cosine",
        "warmup_epochs": 2,
        "is_2d": False
    },
    "E_SwinUNETR": {
        "name": "Swin-UNETR 3D",
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "lr": 1e-4,
        "lr_scheduler": "cosine",
        "warmup_epochs": 2,
        "is_2d": False
    },
    "F_nnUZoo": {
        "name": "nnUZoo / X2Net Hybrid",
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "lr": 1e-4,
        "lr_scheduler": "cosine",
        "warmup_epochs": 2,
        "is_2d": False
    }
}
