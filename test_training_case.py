import torch
import numpy as np
from models.fold_runner import load_case_data, validate_model
from models.model_factory import get_model
from models.common_config import PATCH_SIZE, ARCH_CONFIGS, STAGE1_FIXED_VAL_CASES, BENCHMARK_RESULTS_DIR
import os

device = torch.device("cuda:0")
arch_key = "A_nnUNet"
ckpt_path = os.path.join(BENCHMARK_RESULTS_DIR, 'stage1_screening', 'A_nnUNet', 'checkpoint_best.pth')
model = get_model(arch_key).to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device))

# Pick a training case (e.g., first in the training list)
# We need to know a training case ID from splits_final.json. You can hardcode one.
case_id = "colon_039"  # This is a validation case, but we can test any.
image_arr, label_arr = load_case_data(case_id)
lbl = label_arr[0]
image_tensor = torch.from_numpy(image_arr).float().to(device)

from models.evaluation import run_sliding_window
with torch.no_grad():
    logits = run_sliding_window(model, image_tensor, patch_size=PATCH_SIZE, stride=0.75, device=device)
    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    print("Prediction unique:", np.unique(pred))
    print("Foreground voxels:", np.sum(pred))
    # Also check logits stats for class 1
    logits_np = logits.cpu().numpy()[0]  # (2, Z, Y, X)
    print("Logits class1 max:", logits_np[1].max())
    print("Logits class1 mean:", logits_np[1].mean())