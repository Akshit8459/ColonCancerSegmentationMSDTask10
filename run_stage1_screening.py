#!/usr/bin/env python3
"""
run_stage1_screening.py
=============================================================================
Stage 1 Iso-Time Screening Launcher (Fold 0 across selected architectures).
- Supports parallel multi-GPU execution via --archs arguments.
- Runs 25 epochs per model on Fold 0 with 5-epoch validation cadence.
- Validates on the fixed 5-case performance-stratified subset (STAGE1_FIXED_VAL_CASES).
- Applies Epoch-15 Early Termination Floor (Dice < 0.45).
- Ranks all architectures and outputs stage1_screening_rankings.json.
"""

import os
import sys
import json
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from models.common_config import (
    ARCH_CONFIGS, SPLITS_FINAL_PATH, STAGE1_FIXED_VAL_CASES,
    BENCHMARK_RESULTS_DIR, STAGE2_MAX_FINALISTS
)
from models.fold_runner import run_fold_training

def main():
    parser = argparse.ArgumentParser(description="Stage 1 Iso-Time Screening Launcher")
    parser.add_argument('--archs', nargs='+', default=list(ARCH_CONFIGS.keys()), help="Architectures to screen (e.g. A_nnUNet B_UMamba)")
    args = parser.parse_args()

    print("======================================================================")
    print(" 🚀 STAGE 1: SCREENING (FOLD 0)")
    print(f" Target Architectures: {args.archs}")
    print(" Target: Screen models, apply Epoch-15 Early Termination Floor (<0.45)")
    print(" Fixed Validation Subset: 5 stratified cases in Fold 0")
    print("======================================================================\n", flush=True)

    splits = json.load(open(SPLITS_FINAL_PATH))
    fold0_train = splits[0]['train']
    fold0_val = STAGE1_FIXED_VAL_CASES  # Fixed stratified 5-case subset

    results = {}

    for arch_key in args.archs:
        if arch_key not in ARCH_CONFIGS:
            print(f" ⚠️ Skipping unknown architecture: {arch_key}")
            continue

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

    # Consolidate all available Stage 1 results across models
    stage1_dir = os.path.join(BENCHMARK_RESULTS_DIR, 'stage1_screening')
    all_results = {}
    for arch in sorted(os.listdir(stage1_dir)):
        summary_path = os.path.join(stage1_dir, arch, 'fold_summary.json')
        if os.path.exists(summary_path):
            all_results[arch] = json.load(open(summary_path))

    if not all_results:
        print(" ⚠️ No completed Stage 1 runs found to rank.")
        return

    # Ranking and Promotion Selection
    print("\n======================================================================")
    print(" 📊 STAGE 1 SCREENING RANKING & SELECTION SUMMARY")
    print("======================================================================")
    
    ranked_models = sorted(all_results.items(), key=lambda x: x[1]["best_val_dice"], reverse=True)
    
    for rank, (key, res) in enumerate(ranked_models, start=1):
        status = "EARLY TERMINATED" if res.get("early_terminated", False) else "COMPLETED"
        print(f" #{rank}: {key:15s} | Best Val Dice: {res['best_val_dice']:.4f} | Status: {status} | Time: {res['total_elapsed_seconds']:.1f}s")

    # Select Winner + Mandatory Exp A Anchor
    top_winner_key = ranked_models[0][0]
    promoted_finalists = ["A_nnUNet"]
    if top_winner_key != "A_nnUNet":
        promoted_finalists.append(top_winner_key)
    else:
        if len(ranked_models) > 1:
            promoted_finalists.append(ranked_models[1][0])

    print("\n======================================================================")
    print(f" 🏆 STAGE 1 WINNER   : {top_winner_key} (Dice: {all_results[top_winner_key]['best_val_dice']:.4f})")
    print(f" 🎟️ PROMOTED TO STAGE 2: {promoted_finalists}")
    print("======================================================================\n", flush=True)

    ranking_output = {
        "rankings": [
            {
                "rank": i + 1,
                "arch_key": k,
                "best_val_dice": v["best_val_dice"],
                "early_terminated": v.get("early_terminated", False),
                "total_elapsed_seconds": v["total_elapsed_seconds"]
            }
            for i, (k, v) in enumerate(ranked_models)
        ],
        "top_winner": top_winner_key,
        "promoted_finalists": promoted_finalists
    }

    from models.plotting import plot_stage1_curves
    plot_stage1_curves(stage1_dir, BENCHMARK_RESULTS_DIR)

    report_path = os.path.join(BENCHMARK_RESULTS_DIR, 'stage1_screening_rankings.json')
    with open(report_path, 'w') as f:
        json.dump(ranking_output, f, indent=4)
        
    print(f" Saved Stage 1 rankings to: {report_path}\n")

if __name__ == '__main__':
    main()
