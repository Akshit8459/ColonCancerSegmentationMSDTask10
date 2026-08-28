import torch
from models.model_factory import get_model
from models.common_config import PATCH_SIZE, ARCH_CONFIGS

device = torch.device("cuda:0")
for arch_key in ARCH_CONFIGS.keys():
    model = get_model(arch_key).to(device)
    model.train()
    # Test batch sizes: 1, 2, 4, 8 (stop on OOM)
    for bs in [1, 2, 4, 8]:
        try:
            dummy = torch.randn(bs, 1, *PATCH_SIZE, device=device)
            with torch.amp.autocast('cuda'):
                out = model(dummy)
                loss = out.sum()  # dummy loss
            loss.backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"{arch_key:15s} | batch {bs}: {peak:.2f} GB")
            torch.cuda.reset_peak_memory_stats()
        except RuntimeError as e:
            print(f"{arch_key:15s} | batch {bs}: OOM")
            break
    del model
    torch.cuda.empty_cache()