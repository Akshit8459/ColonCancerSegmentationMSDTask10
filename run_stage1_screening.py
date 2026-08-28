#!/usr/bin/env python3
"""
run_stage1_screening.py
=============================================================================
Stage 1 Iso-Time Screening Launcher (Fold 0 across all 6 architectures).
- Runs 25 epochs per model on Fold 0 with 5-epoch validation cadence.
- Validates on the fixed 5-case performance-stratified subset (STAGE1_FIXED_VAL_CASES).
- Applies Epoch-15 Early Termination Floor (Dice < 0.45).
- Ranks all 6 architectures, applies 1 SD promotion threshold, and selects
  Exp A (Anchor) + Stage 1 Winner for Stage 2 confirmation.
- Output: /home/akshitp/Benchmarking/results/stage1_screening_rankings.json
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, '/home/akshitp/Benchmarking')

from models.common_config import (
    ARCH_CONFIGS, SPLITS_FINAL_PATH, STAGE1_FIXED_VAL_CASES,
    BENCHMARK_RESULTS_DIR, STAGE2_MAX_FINALISTS
)
from models.fold_runner import run_fold_training

def main():
    print("======================================================================")
    print(" 🚀 STAGE 1: ISO-TIME SCREENING (FOLD 0, ALL 6 ARCHITECTURES)")
    print(" Target: Screen all 6 models, apply Epoch-15 Early Termination Floor (<0.45)")
    print(" Fixed Validation Subset: 5 stratified cases in Fold 0")
    print("======================================================================\n", flush=True)

    splits = json.load(open(SPLITS_FINAL_PATH))
    fold0_train = splits[0]['train']
    fold0_val = STAGE1_FIXED_VAL_CASES  # Fixed stratified 5-case subset

    results = {}

    for arch_key in ARCH_CONFIGS.keys():
        print(f"\n======================================================================")
        print(f" ▶️ STAGE 1 SCREENING: {arch_key} ({ARCH_CONFIGS[arch_key]['name']})")
        print(f"======================================================================", flush=True)
        
        output_dir = os.path.join(BENCHMARK_RESULTS_DIR, 'stage1_screening', arch_key)
        summary = run_fold_training(
            arch_key=arch_key,
            fold_idx=0,
            train_cases=fold0_train,
            val_cases=fold0_val,
            output_dir=output_dir,
            is_stage1=True
        )
        results[arch_key] = summary

    # Ranking and Promotion Selection
    print("\n======================================================================")
    print(" 📊 STAGE 1 SCREENING RANKING & SELECTION SUMMARY")
    print("======================================================================")
    
    ranked_models = sorted(results.items(), key=lambda x: x[1]["best_val_dice"], reverse=True)
    
    for rank, (key, res) in enumerate(ranked_models, start=1):
        status = "EARLY TERMINATED" if res["early_terminated"] else "COMPLETED"
        print(f" #{rank}: {key:15s} | Best Val Dice: {res['best_val_dice']:.4f} | Status: {status} | Time: {res['total_elapsed_seconds']:.1f}s")

    # Select Winner + Mandatory Exp A Anchor
    top_winner_key = ranked_models[0][0]
    promoted_finalists = ["A_nnUNet"]
    if top_winner_key != "A_nnUNet":
        promoted_finalists.append(top_winner_key)
    else:
        # If A_nnUNet is top 1, pick top 2 as challenger
        if len(ranked_models) > 1:
            promoted_finalists.append(ranked_models[1][0])

    print("\n======================================================================")
    print(f" 🏆 STAGE 1 WINNER   : {top_winner_key} (Dice: {results[top_winner_key]['best_val_dice']:.4f})")
    print(f" 🎟️ PROMOTED TO STAGE 2: {promoted_finalists}")
    print("======================================================================\n", flush=True)

    ranking_output = {
        "rankings": [
            {
                "rank": i + 1,
                "arch_key": k,
                "best_val_dice": v["best_val_dice"],
                "early_terminated": v["early_terminated"],
                "total_elapsed_seconds": v["total_elapsed_seconds"]
            }
            for i, (k, v) in enumerate(ranked_models)
        ],
        "top_winner": top_winner_key,
        "promoted_finalists": promoted_finalists
    }

    report_path = os.path.join(BENCHMARK_RESULTS_DIR, 'stage1_screening_rankings.json')
    with open(report_path, 'w') as f:
        json.dump(ranking_output, f, indent=4)
        
    print(f" Saved Stage 1 rankings to: {report_path}\n")

if __name__ == '__main__':
    main()
