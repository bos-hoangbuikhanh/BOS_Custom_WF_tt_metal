import datetime
import os

import cv2
import numpy as np
import pytest
import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config
from detectron2.utils.visualizer import ColorMode, Visualizer
from loguru import logger

from models.bos_model.pdl.reference.modelling_utils import DefaultPredictor

# Global scope
CONFIG_FILE = "models/bos_model/pdl/reference/configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml"
WEIGHTS_PATH = "models/bos_model/pdl/reference/pdl_weights.pt"
OUTPUT_DIR = "models/bos_model/pdl/demo/runs/torch_model"


# ---------------------- Configuration Setup ----------------------
def setup_cfg(config_file, weights_path, output_cfg_path=None, new_res=(256, 512)):
    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)

    cfg.merge_from_file(config_file)

    new_height, new_width = new_res

    cfg.INPUT.MIN_SIZE_TEST = new_height
    cfg.INPUT.MAX_SIZE_TEST = new_width

    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.TYPE = "absolute"
    cfg.INPUT.CROP.SIZE = (new_height, new_width)

    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()
    return cfg


# ---------------------- Inference Runner ----------------------
def run_inference_on_image(
    image_path, config_file, weights_path, output_dir="models/bos_model/pdl/demo/runs/torch_model"
):
    os.makedirs(output_dir, exist_ok=True)

    # Prepare modified config path
    modified_config_path = os.path.join(output_dir, "modified_panoptic_deeplab_config.yaml")
    cfg = setup_cfg(config_file, weights_path, output_cfg_path=modified_config_path)
    predictor = DefaultPredictor(cfg)

    # Load and resize image
    im = cv2.imread(image_path)
    if im is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    original_h, original_w = im.shape[:2]
    target_height, target_width = 256, 512
    im_resized = cv2.resize(im, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # Run inference
    outputs = predictor(im_resized)
    panoptic_seg_tensor, segments_info = outputs["panoptic_seg"]

    # Resize segmentation map back to original size
    panoptic_seg_np = panoptic_seg_tensor.to("cpu").numpy()
    panoptic_seg_resized = cv2.resize(
        panoptic_seg_np.astype(np.float32), (original_w, original_h), interpolation=cv2.INTER_NEAREST
    ).astype(np.int32)

    panoptic_seg_for_visualizer = torch.from_numpy(panoptic_seg_resized)

    # Visualization
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    visualizer = Visualizer(im[:, :, ::-1], metadata, scale=1.0, instance_mode=ColorMode.IMAGE)
    out = visualizer.draw_panoptic_seg_predictions(panoptic_seg_for_visualizer, segments_info)
    result_image = out.get_image()[:, :, ::-1]

    # Save result
    base_name = os.path.basename(image_path)
    name_wo_ext = os.path.splitext(base_name)[0]

    # Add timestamp to filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{name_wo_ext}_panoptic_torch_{timestamp}.jpg")
    cv2.imwrite(output_path, result_image)

    logger.info(f"[✔] Inference result saved to: {output_path}")


@pytest.mark.parametrize(
    "input_image_path",
    [
        ("models/bos_model/pdl/data/images//image1.png"),
        ("models/bos_model/pdl/data/images//image2.png"),
        ("models/bos_model/pdl/data/images/image3.png"),
        ("models/bos_model/pdl/data/images/image4.png"),
    ],
)
def test_demo_tt(input_image_path):
    run_inference_on_image(input_image_path, CONFIG_FILE, WEIGHTS_PATH, output_dir=OUTPUT_DIR)
    logger.info("TORCH PIPELINE RUN COMPLETED SUCCESSFULLY")
