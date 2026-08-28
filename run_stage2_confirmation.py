#!/usr/bin/env python3
"""
run_stage2_confirmation.py
=============================================================================
Stage 2 Full 5-Fold Confirmation Launcher.
Runs 5-fold cross-validation for Stage 1 promoted finalists (Exp A + Stage 1 Winner).
- 25 epochs per fold with 5-epoch validation cadence.
- Saves best-epoch checkpoints (checkpoint_best.pth).
- Automatically exports continuous out-of-fold probability maps (.npz).
"""

import os
import sys
import json

sys.path.insert(0, '/home/akshitp/Benchmarking')

from models.common_config import SPLITS_FINAL_PATH, BENCHMARK_RESULTS_DIR
from models.fold_runner import run_fold_training
from models.probability_export import export_model_probabilities

def main():
    print("======================================================================")
    print(" 🚀 STAGE 2: FULL 5-FOLD CONFIRMATION (TOP FINALISTS + BASELINE A)")
    print("======================================================================\n", flush=True)

    ranking_path = os.path.join(BENCHMARK_RESULTS_DIR, 'stage1_screening_rankings.json')
    if not os.path.exists(ranking_path):
        print(f" ❌ Error: Stage 1 ranking file {ranking_path} not found. Run Stage 1 first!")
        sys.exit(1)

    ranking_data = json.load(open(ranking_path))
    finalists = ranking_data.get("promoted_finalists", ["A_nnUNet"])
    splits = json.load(open(SPLITS_FINAL_PATH))

    print(f" Promoted Finalists for Stage 2 5-Fold CV: {finalists}\n", flush=True)

    for arch_key in finalists:
        print(f"\n======================================================================")
        print(f" 🏋️ STARTING 5-FOLD CV: {arch_key}")
        print(f"======================================================================", flush=True)
        
        arch_stage2_dir = os.path.join(BENCHMARK_RESULTS_DIR, 'stage2_confirmation', arch_key)
        
        for fold in range(5):
            train_cases = splits[fold]['train']
            val_cases = splits[fold]['val']
            
            fold_dir = os.path.join(arch_stage2_dir, f"fold_{fold}")
            run_fold_training(
                arch_key=arch_key,
                fold_idx=fold,
                train_cases=train_cases,
                val_cases=val_cases,
                output_dir=fold_dir,
                is_stage1=False
            )
            
            # Export continuous probability maps for held-out fold validation
            ckpt_path = os.path.join(fold_dir, 'checkpoint_best.pth')
            prob_dir = os.path.join(arch_stage2_dir, f"probabilities_fold_{fold}")
            export_model_probabilities(
                arch_key=arch_key,
                checkpoint_path=ckpt_path,
                case_ids=val_cases,
                output_dir=prob_dir
            )

    print("\n======================================================================")
    print(" 🎉 STAGE 2 5-FOLD CONFIRMATION TRAININGS COMPLETE!")
    print("======================================================================\n", flush=True)

if __name__ == '__main__':
    main()
