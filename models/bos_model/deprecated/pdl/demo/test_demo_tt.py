import datetime
import os
from time import time

import cv2
import numpy as np
import pytest
import torch

# The below dependencies are preprocessing and post processing
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config
from detectron2.utils.visualizer import ColorMode, Visualizer
from loguru import logger
from torch.nn import functional as F

import ttnn
from models.bos_model.pdl.reference.norm_utils import BitMasks, ImageList, Instances
from models.bos_model.pdl.reference.panoptic_seg import PanopticDeepLab
from models.bos_model.pdl.reference.post_processing import (
    ResizeShortestEdge,
    get_panoptic_segmentation,
    sem_seg_postprocess,
)
from models.bos_model.pdl.tt.model_processing import create_pdl_model_parameters
from models.bos_model.pdl.tt.ttnn_panoptic_seg import TtPanopticDeepLab

# Global scope
CONFIG_FILE = "models/bos_model/pdl/reference/configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml"
WEIGHTS_PATH = "models/bos_model/pdl/reference/pdl_weights.pt"
OUTPUT_DIR = "models/bos_model/pdl/demo/runs/bos_model"


def preprocessing(cfg, batched_inputs):
    pixel_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
    pixel_std = torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)

    images = [x["image"] for x in batched_inputs]
    images = [(x - pixel_mean) / pixel_std for x in images]
    # To avoid error in ASPP layer when input has different size.
    size_divisibility = 0
    images = ImageList.from_tensors(images, size_divisibility)
    return images


def postprocessing(cfg, batched_inputs, outputs, images):
    # Permute TTNN outputs back to torch shape
    sem_seg_results = ttnn.to_torch(outputs["sem_seg_results"])
    center_results = ttnn.to_torch(outputs["center_results"])
    offset_results = ttnn.to_torch(outputs["offset_results"])

    meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    stuff_area = cfg.MODEL.PANOPTIC_DEEPLAB.STUFF_AREA
    threshold = cfg.MODEL.PANOPTIC_DEEPLAB.CENTER_THRESHOLD
    nms_kernel = cfg.MODEL.PANOPTIC_DEEPLAB.NMS_KERNEL
    top_k = cfg.MODEL.PANOPTIC_DEEPLAB.TOP_K_INSTANCE
    predict_instances = cfg.MODEL.PANOPTIC_DEEPLAB.PREDICT_INSTANCES
    assert (
        cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV == cfg.MODEL.PANOPTIC_DEEPLAB.USE_DEPTHWISE_SEPARABLE_CONV
    )

    processed_results = []
    for sem_seg_result, center_result, offset_result, input_per_image, image_size in zip(
        sem_seg_results, center_results, offset_results, batched_inputs, images.image_sizes
    ):
        height = input_per_image.get("height")
        width = input_per_image.get("width")
        r = sem_seg_postprocess(sem_seg_result, image_size, height, width)
        c = sem_seg_postprocess(center_result, image_size, height, width)
        o = sem_seg_postprocess(offset_result, image_size, height, width)
        # Post-processing to get panoptic segmentation.
        panoptic_image, _ = get_panoptic_segmentation(
            r.argmax(dim=0, keepdim=True),
            c,
            o,
            thing_ids=meta.thing_dataset_id_to_contiguous_id.values(),
            label_divisor=meta.label_divisor,
            stuff_area=stuff_area,
            void_label=-1,
            threshold=threshold,
            nms_kernel=nms_kernel,
            top_k=top_k,
        )
        # For semantic segmentation evaluation.
        processed_results.append({"sem_seg": r})
        panoptic_image = panoptic_image.squeeze(0)
        semantic_prob = F.softmax(r, dim=0)
        # For panoptic segmentation evaluation.
        processed_results[-1]["panoptic_seg"] = (panoptic_image, None)
        # For instance segmentation evaluation.
        if predict_instances:
            instances = []
            panoptic_image_cpu = panoptic_image.cpu().numpy()
            for panoptic_label in np.unique(panoptic_image_cpu):
                if panoptic_label == -1:
                    continue
                pred_class = panoptic_label // meta.label_divisor
                isthing = pred_class in list(meta.thing_dataset_id_to_contiguous_id.values())
                # Get instance segmentation results.
                if isthing:
                    instance = Instances((height, width))
                    # Evaluation code takes continuous id starting from 0
                    instance.pred_classes = torch.tensor([pred_class], device=panoptic_image.device)
                    mask = panoptic_image == panoptic_label
                    instance.pred_masks = mask.unsqueeze(0)
                    # Average semantic probability
                    sem_scores = semantic_prob[pred_class, ...]
                    sem_scores = torch.mean(sem_scores[mask])
                    # Center point probability
                    mask_indices = torch.nonzero(mask).float()
                    center_y, center_x = (
                        torch.mean(mask_indices[:, 0]),
                        torch.mean(mask_indices[:, 1]),
                    )
                    center_scores = c[0, int(center_y.item()), int(center_x.item())]
                    # Confidence score is semantic prob * center prob.
                    instance.scores = torch.tensor([sem_scores * center_scores], device=panoptic_image.device)
                    # Get bounding boxes
                    instance.pred_boxes = BitMasks(instance.pred_masks).get_bounding_boxes()
                    instances.append(instance)
            if len(instances) > 0:
                processed_results[-1]["instances"] = Instances.cat(instances)

        return processed_results


def preprate_ttnn_inputs(device, input_tensor):
    in_n, in_c, in_h, in_w = input_tensor.shape
    ttnn_input = input_tensor.permute((0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, device=device)
    ttnn_input = ttnn.reshape(ttnn_input, (1, 1, in_n * in_h * in_w, in_c))

    return ttnn_input


class TtDefaultPredictor:
    def __init__(self, device, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model

        # Setup Torch model
        self.torch_model = PanopticDeepLab(self.cfg)
        self.torch_model.eval()
        state_dict = torch.load(cfg.MODEL.WEIGHTS, map_location="cpu")
        self.torch_model.load_state_dict(state_dict)
        self.torch_model.eval()

        # Setup Tt model
        self.device = device
        parameters = create_pdl_model_parameters(self.torch_model)
        self.tt_model = TtPanopticDeepLab(self.device, parameters, parameters["model_args"])

        self.aug = ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        with torch.no_grad():
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            image.to(self.cfg.MODEL.DEVICE)

            inputs = [{"image": image, "height": height, "width": width}]

            logger.info("Preprocessing.....")
            input_image = preprocessing(self.cfg, inputs)
            ttnn_input = preprate_ttnn_inputs(self.device, input_image.tensor)

            logger.info("Running Inference using TTNN.....")
            outputs = self.tt_model(ttnn_input, inp_h=256, inp_w=512)

            logger.info("Postprocessing.....")
            predictions = postprocessing(self.cfg, inputs, outputs, images=input_image)[0]
            return predictions


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
    device, image_path, config_file, weights_path, output_dir="models/bos_model/pdl/demo/runs/bos_model"
):
    os.makedirs(output_dir, exist_ok=True)

    # Prepare modified config path
    modified_config_path = os.path.join(output_dir, "modified_panoptic_deeplab_config.yaml")
    cfg = setup_cfg(config_file, weights_path, output_cfg_path=modified_config_path)
    predictor = TtDefaultPredictor(device, cfg)

    # Load and resize image
    im = cv2.imread(image_path)
    if im is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    original_h, original_w = im.shape[:2]
    target_height, target_width = 256, 512
    im_resized = cv2.resize(im, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # Run inference
    start_time = time()
    outputs = predictor(im_resized)
    elapsed_time = time() - start_time

    print(f"[0] Time taken = {elapsed_time:.4f} s")
    print(f"[0] FPS = {(1 / (elapsed_time)):.2f} Hz")

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

    print("Final FPS = ", 1 / (time() - start_time))

    # Save result
    base_name = os.path.basename(image_path)
    name_wo_ext = os.path.splitext(base_name)[0]
    # Add timestamp to filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = os.path.join(output_dir, f"{name_wo_ext}_output.jpg")
    cv2.imwrite(output_path, result_image)

    logger.info(f"[✔] Inference result saved to: {output_path}")


# @pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}, {"l1_small_size": 10240 * 6}], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 10240 * 6}], indirect=True)
@pytest.mark.parametrize(
    "input_image_path",
    [
        ("models/bos_model/pdl/data/images/image1.png"),
        ("models/bos_model/pdl/data/images/image2.png"),
        ("models/bos_model/pdl/data/images/image3.png"),
        ("models/bos_model/pdl/data/images/image4.png"),
        ("models/bos_model/pdl/data/images/image5.png"),
        ("models/bos_model/pdl/data/images/image6.png"),
        ("models/bos_model/pdl/data/images/image7.png"),
        ("models/bos_model/pdl/data/images/image8.png"),
        ("models/bos_model/pdl/data/images/image9.png"),
        ("models/bos_model/pdl/data/images/image10.png"),
        ("models/bos_model/pdl/data/images/image11.png"),
    ],
)
def test_demo_tt(device, input_image_path):
    device.enable_program_cache()
    ttnn.device.EnablePersistentKernelCache()
    run_inference_on_image(device, input_image_path, CONFIG_FILE, WEIGHTS_PATH, output_dir=OUTPUT_DIR)

    logger.info("TTNN PIPELINE RUN COMPLETED SUCCESSFULLY")
