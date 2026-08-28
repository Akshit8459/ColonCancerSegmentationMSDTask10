#!/usr/bin/env python3
"""
benchmark_throughput.py
=============================================================================
Day 0 Throughput & Fallback-Resolved Benchmarking Script.
Measures real per-architecture throughput (sec/iter), VRAM usage, and resolves
fallback tiers (Tier 1 vs Tier 2 vs Tier 3) on the target V100 GPU before any training runs.

Outputs: /home/akshitp/Benchmarking/results/throughput_report.json
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn

sys.path.insert(0, '/home/akshitp/Benchmarking')

from models.common_config import ARCH_CONFIGS, PATCH_SIZE, BENCHMARK_RESULTS_DIR, FORCE_FALLBACK_TIER
from models.model_factory import get_model

def benchmark_architecture(arch_key, device):
    print(f"\n======================================================================")
    print(f" ⏱️ BENCHMARKING THROUGHPUT: {arch_key} ({ARCH_CONFIGS[arch_key]['name']})")
    print(f"======================================================================", flush=True)

    arch_cfg = ARCH_CONFIGS[arch_key]
    
    # Try fallback tiers: start from FORCE_FALLBACK_TIER (Tier 2) or Tier 1
    tier_candidates = [FORCE_FALLBACK_TIER] if FORCE_FALLBACK_TIER else [1, 2, 3]
    
    resolved_tier = None
    resolved_sec_per_iter = None
    peak_vram = None
    micro_batch = 1
    grad_accum = 2
    
    for tier in tier_candidates:
        print(f" Testing Tier {tier} config...", flush=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        
        try:
            model = get_model(arch_key).to(device)
            model.train()
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) if arch_cfg["optimizer"] == "AdamW" else torch.optim.SGD(model.parameters(), lr=0.01)
            scaler = torch.cuda.amp.GradScaler(enabled=True)
            
            mb = 2 if tier == 1 else 1
            x = torch.randn((mb, 1, *PATCH_SIZE), device=device)
            y = torch.randint(0, 2, (mb, *PATCH_SIZE), device=device, dtype=torch.long)
            criterion = nn.CrossEntropyLoss()
            
            # Warm-up 10 iterations
            for _ in range(10):
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=True):
                    out = model(x)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            torch.cuda.synchronize()
            
            # Timed 40 iterations
            start_time = time.time()
            num_iters = 40
            for _ in range(num_iters):
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=True):
                    out = model(x)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            sec_per_iter = elapsed / num_iters
            peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3) # GB
            
            resolved_tier = tier
            resolved_sec_per_iter = sec_per_iter
            micro_batch = mb
            grad_accum = 1 if tier == 1 else 2
            
            print(f" ✅ Tier {tier} SUCCESS! Speed: {sec_per_iter:.4f} s/iter | Peak VRAM: {peak_vram:.2f} GB", flush=True)
            break
        except RuntimeError as e:
            print(f" ❌ Tier {tier} FAILED: {e}", flush=True)
            torch.cuda.empty_cache()
            
    return {
        "arch_key": arch_key,
        "name": arch_cfg["name"],
        "resolved_tier": resolved_tier,
        "micro_batch_size": micro_batch,
        "grad_accum_steps": grad_accum,
        "sec_per_iter": float(resolved_sec_per_iter) if resolved_sec_per_iter else 0.0,
        "peak_vram_gb": float(peak_vram) if peak_vram else 0.0,
        "fits_batch2": (resolved_tier == 1)
    }

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("======================================================================")
    print(f" 🚀 DAY 0: PER-ARCHITECTURE THROUGHPUT & VRAM FALLBACK BENCHMARK")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("======================================================================\n", flush=True)
    
    results = {}
    for arch_key in ARCH_CONFIGS.keys():
        res = benchmark_architecture(arch_key, device)
        results[arch_key] = res
        
    report_path = os.path.join(BENCHMARK_RESULTS_DIR, 'throughput_report.json')
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n======================================================================")
    print(f" 🎉 BENCHMARK COMPLETE! Report saved to: {report_path}")
    print("======================================================================")
    for k, v in results.items():
        print(f" • {k:15s} | Tier {v['resolved_tier']} | {v['sec_per_iter']:.4f} s/iter | {v['peak_vram_gb']:.2f} GB VRAM")
    print("======================================================================\n", flush=True)

if __name__ == '__main__':
    main()
