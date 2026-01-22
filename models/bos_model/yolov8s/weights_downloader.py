import os
import pickle
import re

import torch
from ultralytics import YOLO

curr_file_path = os.path.dirname(os.path.realpath(__file__))

# Load YOLOv8s
model = YOLO(os.path.join(curr_file_path, "yolov8s.pt"))
# model.eval()

# Get state_dict
state_dict = model.model.state_dict()

# Optional: Convert to CPU and float32 for portability
# clean_dict = {k: v.cpu().float() for k, v in state_dict.items()}
clean_dict = {}
for k, v in state_dict.items():
    # print(k)
    if ("cv1." in k) and ("m." not in k) and ("9." not in k):
        k_a = re.sub(r"(cv1\.)", r"cv1_a.", k)
        k_b = re.sub(r"(cv1\.)", r"cv1_b.", k)
        if "conv.weight" in k_a:
            clean_dict.update({k_a: v.cpu().float()[: v.shape[0] // 2, :, :, :]})
            clean_dict.update({k_b: v.cpu().float()[v.shape[0] // 2 :, :, :, :]})
        elif (
            ("conv.bias" in k_a)
            or ("bn.weight" in k_a)
            or ("bn.bias" in k_a)
            or ("bn.running_mean" in k_a)
            or ("bn.running_var" in k_a)
        ):
            clean_dict.update({k_a: v.cpu().float()[: v.shape[0] // 2]})
            clean_dict.update({k_b: v.cpu().float()[v.shape[0] // 2 :]})
        elif "bn.num_batches_tracked" in k_a:
            clean_dict.update({k_a: v.cpu().float()})
            clean_dict.update({k_b: v.cpu().float()})
    else:
        clean_dict.update({k: v.cpu().float()})

# Print summary
print(f"Total parameters: {len(clean_dict)}")
print("Sample entries:")
for i, (k, v) in enumerate(clean_dict.items()):
    print(f"{i+1:3d}. {k} : {tuple(v.shape)}")
    if i >= 5:
        break

# Save to file
torch.save(clean_dict, os.path.join(curr_file_path, "yolov8s_weights.pth"))  # Binary file
# OR: Save as a plain Python dictionary for inspection

with open(os.path.join(curr_file_path, "yolov8s_weights.pkl"), "wb") as f:
    pickle.dump(clean_dict, f)

detect_weights = {k.replace("model.22.", ""): v for k, v in state_dict.items() if k.startswith("model.22.")}
torch.save(detect_weights, os.path.join(curr_file_path, "detect_weights.pth"))
