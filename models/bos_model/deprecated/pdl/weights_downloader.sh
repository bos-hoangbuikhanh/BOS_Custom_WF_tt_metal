#!/bin/bash

# Exit if any command fails
set -e

echo "Installing required packages"
# Install PyTorch (CPU-only)
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio

# Install common dependencies
pip install ninja opencv-python matplotlib

# Install panopticapi
pip install git+https://github.com/cocodataset/panopticapi.git

# Install Detectron2
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# Paths
MODEL_DIR="models/bos_model/pdl/reference"
MODEL_PKL="$MODEL_DIR/model_final_bd324a.pkl"
MODEL_PT="$MODEL_DIR/pdl_weights.pt"

# Download the model if not already present
if [ ! -f "$MODEL_PKL" ]; then
    echo "Downloading model_final_bd324a.pkl..."
    wget -O "$MODEL_PKL" \
      https://dl.fbaipublicfiles.com/detectron2/PanopticDeepLab/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32/model_final_bd324a.pkl
else
    echo "Model file already exists: $MODEL_PKL"
fi

# Convert .pkl to .pt
echo "Converting model to PyTorch format..."
python3 - <<'EOF'
import pickle
import torch
import numpy as np

input_path = "models/bos_model/pdl/reference/model_final_bd324a.pkl"
output_path = "models/bos_model/pdl/reference/pdl_weights.pt"

with open(input_path, "rb") as f:
    data = pickle.load(f)
    state_dict = data['model']

    # Convert NumPy arrays → torch.Tensor
    for key, value in list(state_dict.items()):
        if isinstance(value, np.ndarray):
            state_dict[key] = torch.from_numpy(value)

    # Remove keys not in model
    for k in ["pixel_mean", "pixel_std"]:
        state_dict.pop(k, None)

    torch.save(state_dict, output_path)

print(f"Converted weights saved to: {output_path}")
EOF
