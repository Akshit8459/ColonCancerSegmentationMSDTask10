#!/usr/bin/env python3
"""
run_folds_1_to_4.py
=============================================================================
Launcher script to run Folds 1, 2, 3, 4 for a specified architecture (150 epochs per fold).
Automatically exports OOF probability maps (.npz) for held-out fold validation.
"""

import os
import sys
import json
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from models.common_config import SPLITS_FINAL_PATH, BENCHMARK_RESULTS_DIR
from models.fold_runner import run_fold_training
from models.probability_export import export_model_probabilities

def main():
    parser = argparse.ArgumentParser(description="Run Folds 1-4 for a specified architecture")
    parser.add_argument("--arch", type=str, default="A_nnUNet", help="Architecture key (default: A_nnUNet)")
    args = parser.parse_args()

    arch_key = args.arch
    splits = json.load(open(SPLITS_FINAL_PATH))
    stage2_dir = os.path.join(BENCHMARK_RESULTS_DIR, 'stage2_confirmation', arch_key)

    print("======================================================================")
    print(f" 🚀 RUNNING FOLDS 1 TO 4 (150 EPOCHS EACH) FOR: {arch_key}")
    print("======================================================================\n", flush=True)

    for fold in range(1, 5):  # Folds 1, 2, 3, 4
        print(f"\n======================================================================")
        print(f" 🏋️ STARTING FOLD {fold}/4 (Cases: Train={len(splits[fold]['train'])}, Val={len(splits[fold]['val'])})")
        print(f"======================================================================", flush=True)

        train_cases = splits[fold]['train']
        val_cases = splits[fold]['val']
        fold_dir = os.path.join(stage2_dir, f"fold_{fold}")

        run_fold_training(
            arch_key=arch_key,
            fold_idx=fold,
            train_cases=train_cases,
            val_cases=val_cases,
            output_dir=fold_dir,
            is_stage1=False
        )

        ckpt_path = os.path.join(fold_dir, 'checkpoint_best.pth')
        prob_dir = os.path.join(stage2_dir, f"probabilities_fold_{fold}")
        export_model_probabilities(
            arch_key=arch_key,
            checkpoint_path=ckpt_path,
            case_ids=val_cases,
            output_dir=prob_dir
        )

    print("\n======================================================================")
    print(f" 🎉 COMPLETED FOLDS 1-4 CV FOR {arch_key}!")
    print("======================================================================\n", flush=True)

if __name__ == '__main__':
    main()
