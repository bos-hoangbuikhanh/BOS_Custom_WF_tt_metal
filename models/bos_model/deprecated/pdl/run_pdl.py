import argparse
import datetime
import math
import os
from time import time

import cv2
import numpy as np
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
# CONFIG_FILE = "models/bos_model/pdl/reference/configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml"
# WEIGHTS_PATH = "models/bos_model/pdl/reference/pdl_weights.pt"
# OUTPUT_DIR = "models/bos_model/pdl/demo/runs/bos_model"

deeplab_home_path = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILE = "reference/configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml"
CONFIG_FILE = os.path.join(deeplab_home_path, CONFIG_FILE)
WEIGHTS_PATH = "reference/pdl_weights.pt"
WEIGHTS_PATH = os.path.join(deeplab_home_path, WEIGHTS_PATH)
OUTPUT_DIR = "output"
OUTPUT_DIR = os.path.join(deeplab_home_path, OUTPUT_DIR)


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--num_images", type=int, default=1, help="number of images to process")
    parser.add_argument("--image_height", type=int, default=256, help="height of output processed by bos model")
    parser.add_argument("--image_width", type=int, default=512, help="width of output processed by bos model")
    parser.add_argument("--trace", action="store_true", help="enable trace mode")
    parser.add_argument(
        "--source",
        type=str,
        default=os.path.join(deeplab_home_path, "test_images"),
        help="Video source: webcam index (e.g., 0) or path/URL",
    )
    parser.add_argument("-p", "--persistent_cache", action="store_true", help="enable trace mode")

    args, _ = parser.parse_known_args(argv)
    return args


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
    # sem_seg_results = ttnn.to_torch(outputs["sem_seg_results"])
    # # ttnn.deallocate(outputs["sem_seg_results"])
    # center_results = ttnn.to_torch(outputs["center_results"])
    # # ttnn.deallocate(outputs["center_results"])
    # offset_results = ttnn.to_torch(outputs["offset_results"])
    # # ttnn.deallocate(outputs["offset_results"])

    sem_seg_results = outputs["sem_seg_results"]
    center_results = outputs["center_results"]
    offset_results = outputs["offset_results"]

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


def setup_l1_sharded_input(device, torch_input_tensor=None, min_channels=3, num_cores=20):
    if num_cores == 20:
        core_grid = ttnn.CoreGrid(y=4, x=5)
    elif num_cores == 16:
        core_grid = ttnn.CoreGrid(y=4, x=4)
    else:
        core_grid = ttnn.CoreGrid(y=8, x=8)
    torch_input_tensor = torch_input_tensor.permute((0, 2, 3, 1))  # NCHW -> NHWC
    n, h, w, c = torch_input_tensor.shape
    if c < min_channels:
        channel_padding_needed = min_channels - c
        torch_input_tensor = torch.nn.functional.pad(
            torch_input_tensor, (0, channel_padding_needed, 0, 0, 0, 0), value=0.0
        )
        c = min_channels
    torch_input_tensor = torch_input_tensor.reshape(1, 1, n * h * w, c)
    nhw = n * h * w
    shard_size = math.ceil((nhw / num_cores) / 32) * 32
    input_mem_config = ttnn.create_sharded_memory_config(
        [1, 1, shard_size * num_cores, c],
        core_grid,
        ttnn.ShardStrategy.HEIGHT,
    )
    tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    return tt_inputs_host, input_mem_config


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


def main(device, args):
    images = list(os.path.join(args.source, "image" + str(i + 1) + ".png") for i in range(11))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Prepare modified config path
    modified_config_path = os.path.join(OUTPUT_DIR, "modified_panoptic_deeplab_config.yaml")
    cfg = setup_cfg(CONFIG_FILE, WEIGHTS_PATH, output_cfg_path=modified_config_path)

    torch_model = PanopticDeepLab(cfg)
    torch_model.eval()
    state_dict = torch.load(cfg.MODEL.WEIGHTS, map_location="cpu")
    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    parameters = create_pdl_model_parameters(torch_model)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    model = TtPanopticDeepLab(device, parameters, parameters["model_args"])
    aug = ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
    input_format = cfg.INPUT.FORMAT
    assert input_format in ["RGB", "BGR"], input_format

    tid = 0
    if args.trace:
        print("Warming up for Trace capture")
        # Load and resize image
        im = cv2.imread(images[0])
        if im is None:
            raise FileNotFoundError(f"Could not load image from {images[0]}")

        original_h, original_w = im.shape[:2]
        target_height, target_width = 256, 512
        original_image = cv2.resize(im, (target_width, target_height), interpolation=cv2.INTER_AREA)
        if input_format == "RGB":
            original_image = original_image[:, :, ::-1]
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        image.to(cfg.MODEL.DEVICE)

        logger.info("Preprocessing.....")
        inputs = [{"image": image, "height": original_h, "width": original_w}]
        input_image = preprocessing(cfg, inputs)
        input_image = input_image.tensor  # .reshape(1,1,target_height*target_width,-1)
        # test_image_host, input_mem_config = setup_l1_sharded_input(device, input_image)
        model.input_tensor = preprate_ttnn_inputs(device, input_image)
        # model.input_tensor = test_image_host.to(device, input_mem_config)

        # Warm up
        # ttnn.deallocate(test_image_host)
        # ttnn.reallocate(model.input_tensor)
        outputs = model()

        if input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            original_image = original_image[:, :, ::-1]

        # Capture
        start_time = time()
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        outputs = model()
        ttnn.end_trace_capture(device, tid, cq_id=0)
        elapsed_time = time() - start_time
        print(f"Warmup Time taken = {elapsed_time:.4f} s")
        print(f"Warmup FPS = {(1 / (elapsed_time)):.2f} Hz")

    for i in range(args.num_images):
        model_start_time = time()
        index = i + 1
        image_path = images[i % 11]
        print(f"Processing image {index}/{args.num_images}")
        # print(image_path)

        # Load and resize image
        im = cv2.imread(image_path)
        if im is None:
            raise FileNotFoundError(f"Could not load image from {images[0]}")

        original_h, original_w = im.shape[:2]
        target_height, target_width = 256, 512
        original_image = cv2.resize(im, (target_width, target_height), interpolation=cv2.INTER_AREA)
        if input_format == "RGB":
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        image.to(cfg.MODEL.DEVICE)

        inputs = [{"image": image, "height": height, "width": width}]
        input_image = preprocessing(cfg, inputs)
        # input_image = input_image.tensor #.reshape(1,1,target_height*target_width,-1)

        ttnn_start_time = time()
        if args.trace:
            # model.input_tensor = preprate_ttnn_inputs(device, input_image.tensor)
            test_image_host, input_mem_config = setup_l1_sharded_input(device, input_image.tensor)
            # model.input_tensor = test_image_host.to(device, input_mem_config)
            ttnn.copy_host_to_device_tensor(test_image_host, model.input_tensor, 0)
            # ttnn.deallocate(test_image_host)
        else:
            input_tensor = preprate_ttnn_inputs(device, input_image.tensor)

        if args.trace:
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            # sem_seg_results = model.output0
            # center_results = model.output1
            # offset_results = model.output2
            outputs = {
                "sem_seg_results": ttnn.to_torch(model.output0),
                "center_results": ttnn.to_torch(model.output1),
                "offset_results": ttnn.to_torch(model.output2),
            }
        else:
            outputs = model(input_tensor)
            sem_seg_results = ttnn.to_torch(outputs["sem_seg_results"])
            center_results = ttnn.to_torch(outputs["center_results"])
            offset_results = ttnn.to_torch(outputs["offset_results"])
            outputs = {
                "sem_seg_results": ttnn.to_torch(outputs["sem_seg_results"]),
                "center_results": ttnn.to_torch(outputs["center_results"]),
                "offset_results": ttnn.to_torch(outputs["offset_results"]),
            }

        ttnn.synchronize_device(device)
        ttnn_elapsed_time = time() - ttnn_start_time

        print(f"[{index}] Time taken = {ttnn_elapsed_time:.4f} s")
        print(f"[{index}] FPS = {(1 / (ttnn_elapsed_time)):.2f} Hz")

        outputs = postprocessing(cfg, inputs, outputs, images=input_image)[0]

        panoptic_seg_tensor, segments_info = outputs["panoptic_seg"]

        # Resize segmentation map back to original size
        panoptic_seg_np = panoptic_seg_tensor.to("cpu").numpy()
        panoptic_seg_resized = cv2.resize(
            panoptic_seg_np.astype(np.float32), (original_w, original_h), interpolation=cv2.INTER_NEAREST
        ).astype(np.int32)

        panoptic_seg_for_visualizer = torch.from_numpy(panoptic_seg_resized)

        # Visualization
        visualizer = Visualizer(im[:, :, ::-1], metadata, scale=1.0, instance_mode=ColorMode.IMAGE)
        out = visualizer.draw_panoptic_seg_predictions(panoptic_seg_for_visualizer, segments_info)
        result_image = out.get_image()[:, :, ::-1]

        print("Total FPS = ", 1 / (time() - model_start_time))

        # Save result
        base_name = os.path.basename(image_path)
        name_wo_ext = os.path.splitext(base_name)[0]
        # Add timestamp to filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        output_path = os.path.join(OUTPUT_DIR, f"{name_wo_ext}_output.jpg")
        cv2.imwrite(output_path, result_image)

        logger.info(f"[✔] Inference result saved to: {output_path}")


if __name__ == "__main__":
    l1_small_size = 10240 * 6
    args = parse_args()
    if args.trace:
        device = ttnn.open_device(device_id=0, l1_small_size=l1_small_size, trace_region_size=10419200)
    else:
        device = ttnn.open_device(device_id=0, l1_small_size=l1_small_size)
    device.enable_program_cache()
    if args.persistent_cache:
        ttnn.device.EnablePersistentKernelCache()

    main(device, args)
    ttnn.close_device(device)
