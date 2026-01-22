import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from pathlib import Path

from copy import deepcopy

import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from ultralytics import YOLO


def load_resize_and_pad_channels(image_path, image_size=[320, 320], channels=3):
    # Load and resize to (320, 320)
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.Resize(image_size), T.ToTensor()])
    tensor = transform(img)

    # Pad channels from 3 → 32
    c, h, w = tensor.shape
    if c < channels:
        pad_c = channels - c
        pad_tensor = torch.zeros((pad_c, h, w), dtype=tensor.dtype)
        tensor = torch.cat([tensor, pad_tensor], dim=0)

    # Add batch dimension → [1, 32, 320, 320]
    return tensor.unsqueeze(0), torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0)


# Load the pre-trained YOLOv8s model
model = YOLO("yolov8s.pt")
model.eval()

# Path to your image(s)
images = [
    "16004479832_a748d55f21_k.jpg",
    "17790319373_bd19b24cfc_k.jpg",
    "18124840932_e42b3e377c_k.jpg",
    "19064748793_bb942deea1_k.jpg",
    "24274813513_0cfd2ce6d0_k.jpg",
    "33823288584_1d21cf0a26_k.jpg",
    "33887522274_eebd074106_k.jpg",
]

# Output directory
input_dir = "images"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)


def torch_load_resize_and_pad_channels(image_path, image_size=[320, 320], channels=3):
    # Load and resize to (320, 320)
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.Resize(image_size),
            T.ToTensor(),
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    tensor = transform(img)

    # Pad channels from 3 → 32
    c, h, w = tensor.shape
    if c < channels:
        pad_c = channels - c
        pad_tensor = torch.zeros((pad_c, h, w), dtype=tensor.dtype)
        tensor = torch.cat([tensor, pad_tensor], dim=0)

    # Add batch dimension → [1, 32, 320, 320]
    return tensor.unsqueeze(0)


image_height = 256
image_width = 256
input_channels = 3

state_dict = model.state_dict()
detect_weights = {k.replace("model.model.22.", ""): v for k, v in state_dict.items() if k.startswith("model.model.22.")}
torch.save(detect_weights, "detect_weights.pth")

for name, param in detect_weights.items():
    print(name)

# Inference loop
for image_name in images:
    image, org_image = load_resize_and_pad_channels(os.path.join(input_dir, image_name))

    # Run inference
    results = model(image)

    # Visualize and save the result (bounding boxes drawn)
    for i, result in enumerate(results):
        boxed_img = result.plot()[:, :, ::-1]  # Convert BGR to RGB

        # Save image
        filename = os.path.basename(image_name)
        out_path = os.path.join(output_dir, f"ultra_{filename}")
        cv2.imwrite(out_path, boxed_img)
        print(f"Saved: {out_path}")
