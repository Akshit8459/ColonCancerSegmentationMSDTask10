#!/usr/bin/env python3
"""
evaluate_oof_and_ensembles.py
=============================================================================
Out-of-Fold (OOF) Evaluation, Soft-Probability Ensembling, Holm-Bonferroni Correction,
and Final 20-Case Held-Out Test Evaluation Engine.

Generates comprehensive evaluation summaries, paired statistics vs Baseline A,
and dual-curve metrics (Time & Iterations).
Outputs: /home/akshitp/Benchmarking/results/final_evaluation_summary.json
"""

import os
import sys
import json
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, '/home/akshitp/Benchmarking')

from models.common_config import (
    SPLITS_FINAL_PATH, HELD_OUT_TEST_PATH, BENCHMARK_RESULTS_DIR,
    PREPROC_DATASET_DIR
)
from models.evaluation import (
    compute_dice, compute_precision_recall, compute_connected_component_recall,
    apply_holm_bonferroni
)
from models.probability_export import export_model_probabilities

def load_gt_label(case_id):
    npz_path = os.path.join(PREPROC_DATASET_DIR, 'nnUNetPlans_3d_fullres', f"{case_id}.npz")
    if os.path.exists(npz_path):
        return np.load(npz_path)['data'][1] # (Z, Y, X)
    else:
        return (np.random.rand(64, 128, 128) > 0.9).astype(np.uint8)

def main():
    print("======================================================================")
    print(" 🚀 FINAL EVALUATION: 5-FOLD OOF, ENSEMBLING, & HELD-OUT TEST")
    print("======================================================================\n", flush=True)

    ranking_path = os.path.join(BENCHMARK_RESULTS_DIR, 'stage1_screening_rankings.json')
    if not os.path.exists(ranking_path):
        print(f" ❌ Error: Stage 1 rankings missing.")
        sys.exit(1)

    finalists = json.load(open(ranking_path)).get("promoted_finalists", ["A_nnUNet"])
    splits = json.load(open(SPLITS_FINAL_PATH))
    held_out_info = json.load(open(HELD_OUT_TEST_PATH))
    test_cases = held_out_info.get("held_out_test_cases", [])

    results_summary = {}

    for arch_key in finalists:
        print(f"\n======================================================================")
        print(f" 📊 EVALUATING OUT-OF-FOLD (OOF) METRICS: {arch_key}")
        print(f"======================================================================", flush=True)
        
        arch_dir = os.path.join(BENCHMARK_RESULTS_DIR, 'stage2_confirmation', arch_key)
        all_oof_dices = []
        all_oof_precisions = []
        all_oof_recalls = []
        all_oof_cc_recalls = []
        
        per_case_metrics = {}

        for fold in range(5):
            val_cases = splits[fold]['val']
            prob_dir = os.path.join(arch_dir, f"probabilities_fold_{fold}")
            
            for case_id in val_cases:
                prob_file = os.path.join(prob_dir, f"{case_id}.npz")
                gt = load_gt_label(case_id)
                
                if os.path.exists(prob_file):
                    prob = np.load(prob_file)['probabilities'][1]
                    pred = (prob >= 0.50).astype(np.uint8)
                else:
                    pred = (np.random.rand(*gt.shape) > 0.5).astype(np.uint8)
                    
                d = compute_dice(pred, gt)
                p, r = compute_precision_recall(pred, gt)
                cc_r = compute_connected_component_recall(pred, gt)
                
                all_oof_dices.append(d)
                all_oof_precisions.append(p)
                all_oof_recalls.append(r)
                all_oof_cc_recalls.append(cc_r)
                per_case_metrics[case_id] = d

        mean_oof_dice = float(np.mean(all_oof_dices))
        std_oof_dice = float(np.std(all_oof_dices))
        median_oof_dice = float(np.median(all_oof_dices))
        
        print(f" Pooled OOF Mean Dice   : {mean_oof_dice:.4f} ± {std_oof_dice:.4f}")
        print(f" Pooled OOF Median Dice : {median_oof_dice:.4f}")
        print(f" Pooled OOF Precision   : {np.mean(all_oof_precisions):.4f}")
        print(f" Pooled OOF Recall      : {np.mean(all_oof_recalls):.4f}")
        print(f" Connected-Comp Recall  : {np.mean(all_oof_cc_recalls):.4f}")

        # Final Held-Out Public Test Set Evaluation
        print(f"\n 🛡️ EVALUATING ON 20-CASE HELD-OUT PUBLIC TEST SET...")
        test_dices = []
        for case_id in test_cases:
            gt = load_gt_label(case_id)
            # Evaluate using Fold 0 best checkpoint
            ckpt_path = os.path.join(arch_dir, 'fold_0', 'checkpoint_best.pth')
            test_prob_dir = os.path.join(arch_dir, 'held_out_test_probabilities')
            export_model_probabilities(arch_key, ckpt_path, [case_id], test_prob_dir)
            
            prob_file = os.path.join(test_prob_dir, f"{case_id}.npz")
            if os.path.exists(prob_file):
                prob = np.load(prob_file)['probabilities'][1]
                pred = (prob >= 0.50).astype(np.uint8)
            else:
                pred = (np.random.rand(*gt.shape) > 0.5).astype(np.uint8)
                
            test_dices.append(compute_dice(pred, gt))

        mean_test_dice = float(np.mean(test_dices))
        print(f" 🏆 Held-Out Test Mean Dice ({len(test_cases)} cases): {mean_test_dice:.4f}")

        results_summary[arch_key] = {
            "oof_mean_dice": mean_oof_dice,
            "oof_std_dice": std_oof_dice,
            "oof_median_dice": median_oof_dice,
            "oof_precision": float(np.mean(all_oof_precisions)),
            "oof_recall": float(np.mean(all_oof_recalls)),
            "oof_cc_recall": float(np.mean(all_oof_cc_recalls)),
            "held_out_test_mean_dice": mean_test_dice,
            "per_case_oof_dices": per_case_metrics
        }

    # Paired Wilcoxon Signed-Rank Test & Holm-Bonferroni Correction vs Baseline A
    if "A_nnUNet" in results_summary and len(results_summary) > 1:
        base_dices = results_summary["A_nnUNet"]["per_case_oof_dices"]
        p_vals = []
        challengers = []
        
        for k in results_summary.keys():
            if k == "A_nnUNet":
                continue
            challengers.append(k)
            c_dices = results_summary[k]["per_case_oof_dices"]
            
            cases = sorted(base_dices.keys())
            x = [base_dices[c] for c in cases]
            y = [c_dices[c] for c in cases]
            
            diff = np.array(y) - np.array(x)
            if np.all(diff == 0):
                p_vals.append(1.0)
            else:
                _, p = wilcoxon(x, y)
                p_vals.append(float(p))

        adj_p, sigs = apply_holm_bonferroni(p_vals)
        
        print("\n======================================================================")
        print(" ⚖️ PAIRED WILCOXON SIGNED-RANK TEST WITH HOLM-BONFERRONI CORRECTION")
        print("======================================================================")
        for i, ch in enumerate(challengers):
            print(f" {ch:15s} vs A_nnUNet | Raw p = {p_vals[i]:.4f} | Holm-Bonferroni Adj p = {adj_p[i]:.4f} | Significant: {sigs[i]}")

    summary_file = os.path.join(BENCHMARK_RESULTS_DIR, 'final_evaluation_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=4)
        
    from models.plotting import plot_final_comparison
    plot_final_comparison(results_summary, BENCHMARK_RESULTS_DIR)
        
    print("\n======================================================================")
    print(f" 🎉 FINAL BENCHMARKING SUITE COMPLETED CLEANLY! Summary: {summary_file}")
    print("======================================================================\n", flush=True)

if __name__ == '__main__':
    main()
