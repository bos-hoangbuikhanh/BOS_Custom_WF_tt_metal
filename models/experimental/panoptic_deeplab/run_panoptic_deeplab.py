import argparse
import datetime
import math
import os
from time import time

import cv2
import numpy as np
import torch

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import check_with_pcc
from models.bos_model.panoptic_deeplab.tt.model_preprocessing import (
    create_panoptic_deeplab_parameters,
    fuse_conv_bn_parameters,
)
from models.bos_model.panoptic_deeplab.tt.tt_model import TtPanopticDeepLab
from models.bos_model.panoptic_deeplab.reference.pytorch_model import PytorchPanopticDeepLab
from models.bos_model.panoptic_deeplab.tt.model_configs import ModelOptimisations
from models.bos_model.panoptic_deeplab.tt.common import (
    PDL_L1_SMALL_SIZE,
    get_panoptic_deeplab_weights_path,
    get_panoptic_deeplab_config,
    preprocess_nchw_input_tensor,
)

pdl_home_path = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILE = "reference/configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml"
CONFIG_FILE = os.path.join(pdl_home_path, CONFIG_FILE)
WEIGHTS_PATH = "weights/model_final_bd324a.pkl"
WEIGHTS_PATH = os.path.join(pdl_home_path, WEIGHTS_PATH)
OUTPUT_DIR = "output"
OUTPUT_DIR = os.path.join(pdl_home_path, OUTPUT_DIR)


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--num_images", type=int, default=1, help="number of images to process")
    parser.add_argument("--image_height", type=int, default=256, help="height of output processed by bos model")
    parser.add_argument("--image_width", type=int, default=512, help="width of output processed by bos model")
    parser.add_argument("--trace", action="store_true", help="enable trace mode")
    parser.add_argument("--weights_path", type=str, default=WEIGHTS_PATH, help="Location of model weights file")
    parser.add_argument("-p", "--persistent_cache", action="store_true", help="enable trace mode")
    parser.add_argument("--enable_logger", action="store_true", help="enable logger mode")

    args, _ = parser.parse_known_args(argv)
    return args


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


def main(device, args):
    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"Could not find weights file at {args.weights_path}")

    # images = list(os.path.join(args.source, "image" + str(i + 1) + ".png") for i in range(11))
    # os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get model configuration
    config = get_panoptic_deeplab_config()
    batch_size = config["batch_size"]
    num_classes = config["num_classes"]
    project_channels = config["project_channels"]
    decoder_channels = config["decoder_channels"]
    sem_seg_head_channels = config["sem_seg_head_channels"]
    ins_embed_head_channels = config["ins_embed_head_channels"]
    common_stride = config["common_stride"]
    train_size = config["train_size"]

    input_height, input_width = train_size[0], train_size[1]
    input_channels = 3

    pytorch_model = PytorchPanopticDeepLab(
        num_classes=num_classes,
        common_stride=common_stride,
        project_channels=project_channels,
        decoder_channels=decoder_channels,
        sem_seg_head_channels=sem_seg_head_channels,
        ins_embed_head_channels=ins_embed_head_channels,
        train_size=train_size,
        weights_path=WEIGHTS_PATH,
    )
    pytorch_model = pytorch_model.to(dtype=torch.bfloat16)
    pytorch_model.eval()

    ttnn_parameters = create_panoptic_deeplab_parameters(
        pytorch_model, device, input_height=input_height, input_width=input_width, batch_size=batch_size
    )

    # Apply Conv+BatchNorm fusion to the parameters
    logger.info("Applying Conv+BatchNorm fusion to parameters...")
    fused_parameters = fuse_conv_bn_parameters(ttnn_parameters, eps=1e-5)
    logger.info("Conv+BatchNorm fusion completed successfully")

    # Create centralized configuration
    model_configs = ModelOptimisations(
        conv_act_dtype=ttnn.bfloat8_b,
        conv_w_dtype=ttnn.bfloat8_b,
    )

    # Apply layer-specific configurations
    logger.info("Applying ResNet backbone configurations...")
    model_configs.setup_resnet_backbone()
    logger.info("Applying ASPP layer overrides...")
    model_configs.setup_aspp()
    logger.info("Applying decoder layer overrides...")
    model_configs.setup_decoder()
    logger.info("Applying head layer overrides...")
    model_configs.setup_heads()

    ttnn_model = TtPanopticDeepLab(
        device=device,
        parameters=fused_parameters,
        num_classes=num_classes,
        common_stride=common_stride,
        project_channels=project_channels,
        decoder_channels=decoder_channels,
        sem_seg_head_channels=sem_seg_head_channels,
        ins_embed_head_channels=ins_embed_head_channels,
        train_size=train_size,
        model_configs=model_configs,
    )

    tid = 0
    if args.trace:
        print("Warming up for Trace capture")
        pytorch_input = torch.randn(batch_size, input_channels, input_height, input_width, dtype=torch.bfloat16)
        ttnn_model.input_tensor = preprocess_nchw_input_tensor(device, pytorch_input)

        # ttnn.deallocate(test_image_host)
        # ttnn.reallocate(model.input_tensor)
        outputs = ttnn_model.forward()
        outputs = ttnn_model.forward()

        # Capture
        start_time = time()
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        outputs = ttnn_model.forward()
        ttnn.end_trace_capture(device, tid, cq_id=0)
        elapsed_time = time() - start_time
        print(f"Warmup Time taken = {elapsed_time:.4f} s")
        print(f"Warmup FPS = {(1 / (elapsed_time)):.2f} Hz")

    for i in range(args.num_images):
        index = i + 1
        # image_path = images[i % 11]
        # print(f"Processing image {index}/{args.num_images}")

        # # Load and resize image
        # im = cv2.imread(image_path)
        # if im is None:
        #     raise FileNotFoundError(f"Could not load image from {images[0]}")

        # original_h, original_w = im.shape[:2]
        # target_height, target_width = 256, 512
        # original_image = cv2.resize(im, (target_width, target_height), interpolation=cv2.INTER_AREA)
        # if input_format == "RGB":
        #     original_image = original_image[:, :, ::-1]
        # height, width = original_image.shape[:2]
        # image = aug.get_transform(original_image).apply_image(original_image)
        # image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        # image.to(cfg.MODEL.DEVICE)

        # inputs = [{"image": image, "height": height, "width": width}]
        # input_image = preprocessing(cfg, inputs)
        # # input_image = input_image.tensor #.reshape(1,1,target_height*target_width,-1)

        pytorch_input = torch.randn(batch_size, input_channels, input_height, input_width, dtype=torch.bfloat16)

        ttnn_start_time = time()
        if args.trace:
            test_image_host = preprocess_nchw_input_tensor(device, pytorch_input)
            ttnn_model.input_tensor = test_image_host.to(device, test_image_host.memory_config())
            # ttnn.copy_host_to_device_tensor(test_image_host, ttnn_model.input_tensor, 0)
            # ttnn.deallocate(test_image_host)
        else:
            ttnn_input = preprocess_nchw_input_tensor(device, pytorch_input)

        if args.trace:
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            ttnn_semantic = ttnn_model.output0
            ttnn_center = ttnn_model.output1
            ttnn_offset = ttnn_model.output2
        else:
            ttnn_semantic, ttnn_center, ttnn_offset, _ = ttnn_model.forward(ttnn_input)

        # Handle semantic output - slice back to original channels if padding was applied
        ttnn_semantic_torch = ttnn.to_torch(ttnn_semantic).permute(0, 3, 1, 2)
        semantic_original_channels = ttnn_model.semantic_head.get_output_channels_for_slicing()
        if semantic_original_channels is not None:
            logger.info(
                f"Slicing semantic output from {ttnn_semantic_torch.shape[1]} to {semantic_original_channels} channels"
            )
            ttnn_semantic_torch = ttnn_semantic_torch[:, :semantic_original_channels, :, :]

        # Handle center output - slice back to original channels if padding was applied
        ttnn_center_torch = ttnn.to_torch(ttnn_center).permute(0, 3, 1, 2)
        center_original_channels = ttnn_model.instance_head.get_center_output_channels_for_slicing()
        if center_original_channels is not None:
            logger.info(
                f"Slicing center output from {ttnn_center_torch.shape[1]} to {center_original_channels} channels"
            )
            ttnn_center_torch = ttnn_center_torch[:, :center_original_channels, :, :]

        # Handle offset output - slice back to original channels if padding was applied
        ttnn_offset_torch = ttnn.to_torch(ttnn_offset).permute(0, 3, 1, 2)
        offset_original_channels = ttnn_model.instance_head.get_offset_output_channels_for_slicing()
        if offset_original_channels is not None:
            logger.info(
                f"Slicing offset output from {ttnn_offset_torch.shape[1]} to {offset_original_channels} channels"
            )
            ttnn_offset_torch = ttnn_offset_torch[:, :offset_original_channels, :, :]

        ttnn.synchronize_device(device)
        ttnn_elapsed_time = time() - ttnn_start_time

        print(f"[{index}] Time taken = {ttnn_elapsed_time:.4f} s")
        print(f"[{index}] FPS = {(1 / (ttnn_elapsed_time)):.2f} Hz")

        pytorch_model = pytorch_model.to(dtype=torch.float32)
        pytorch_input = pytorch_input.to(dtype=torch.float32)
        pytorch_semantic, pytorch_center, pytorch_offset, _ = pytorch_model.forward(pytorch_input)

        sem_passed, sem_msg = check_with_pcc(pytorch_semantic, ttnn_semantic_torch, pcc=0.99)
        logger.info(f"Semantic PCC: {sem_msg}")
        print(f"Semantic PCC: {sem_msg}")

        center_passed, center_msg = check_with_pcc(pytorch_center, ttnn_center_torch, pcc=0.99)
        logger.info(f"Center PCC: {center_msg}")
        print(f"Center PCC: {center_msg}")

        offset_passed, offset_msg = check_with_pcc(pytorch_offset, ttnn_offset_torch, pcc=0.99)
        logger.info(f"Offset PCC: {offset_msg}")
        print(f"Offset PCC: {offset_msg}")
        print()

        # input("Press Enter to continue...")


if __name__ == "__main__":
    l1_small_size = 10240 * 6
    args = parse_args()
    if not args.enable_logger:
        logger.remove()
    if args.trace:
        device = ttnn.open_device(device_id=0, l1_small_size=l1_small_size, trace_region_size=10419200)
    else:
        device = ttnn.open_device(device_id=0, l1_small_size=l1_small_size)
    device.enable_program_cache()
    if args.persistent_cache:
        ttnn.device.EnablePersistentKernelCache()

    main(device, args)
    ttnn.close_device(device)
