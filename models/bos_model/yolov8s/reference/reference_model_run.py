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

# from ultralytics import YOLO
from ultralytics.utils import ops
from reference_model import YOLOv8_AlignedWithPT as YOLO

# torch.set_printoptions(precision=2, threshold=100)

class_names = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}


def load_resize_and_pad_channels(image_path, image_size=[320, 320], channels=3):
    # Load and resize to (320, 320)
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.Resize(image_size),
            T.ToTensor(),
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


# Load the pre-trained YOLOv8s model
# model = YOLO('yolov8s.pt')  # Make sure you have this weight file downloaded

model = YOLO()
# print(model)
# for name, param in model.named_parameters():
#     print(name)
model.load_ckpt("yolov8s.pt")
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
    # 'image_1.jpg',
    # 'image_2.jpg',
    # 'image_3.jpg',
    # 'image_4.jpg',
    # 'image_5.jpg',
]

# Output directory
input_dir = "images"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Inference loop
for image_name in images:
    # Run inference
    image = load_resize_and_pad_channels(os.path.join(input_dir, image_name))
    original_image = cv2.imread(os.path.join(input_dir, image_name))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    preds = model(image)

    nms_result = ops.non_max_suppression(preds[0].clone(), conf_thres=0.25, iou_thres=0.7)

    import supervision as sv

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.6, text_padding=1)

    # Visualize and save the result (bounding boxes drawn)
    det = nms_result[0]
    if det is not None and len(det):
        boxes = det[:, :4].detach().cpu().numpy()
        confs = det[:, 4].detach().cpu().numpy()
        class_ids = det[:, 5].detach().cpu().numpy().astype(int)

        orig_h, orig_w, _ = original_image.shape  # original_image read by cv2
        # orig_w, orig_h = original_image.size # original_image read by PIL
        input_w, input_h = image.shape[-2:]
        scale_x, scale_y = orig_w / input_w, orig_h / input_h
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        detections = sv.Detections(xyxy=boxes, confidence=confs, class_id=class_ids)
        labels = [f"{class_names.get(cls, str(cls))} {conf:.2f}" for cls, conf in zip(class_ids, confs)]

        # 박스 시각화
        annotated_image = box_annotator.annotate(scene=original_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        # 저장
        annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        filename = os.path.basename(image_name)
        out_path = os.path.join("results", f"boxed_{filename}")
        cv2.imwrite(out_path, annotated_bgr)
        print(f"✅ Saved result: {out_path}")
