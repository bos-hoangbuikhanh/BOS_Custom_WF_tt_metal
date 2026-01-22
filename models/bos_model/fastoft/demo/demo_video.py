# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import torch
import ttnn
import pytest
import matplotlib.pyplot as plt
import cv2
import numpy as np
from loguru import logger

from ..reference.bbox import visualize_objects
from ..reference.encoder import ObjectEncoder
from ..reference.oftnet import OftNet, OftMode, FrontendMode
from ..reference.utils import (
    load_calib,
    make_grid,
)
from ..tests.common import (
    GRID_HEIGHT,
    GRID_RES,
    GRID_SIZE,
    H_PADDED,
    NMS_THRESH,
    W_PADDED,
    Y_OFFSET,
    load_checkpoint,
)
from ..tt.model_preprocessing import (
    create_OFT_model_parameters,
    create_decoder_model_parameters,
    fuse_conv_bn_parameters,
    fuse_imagenet_normalization,
)
from ..tt.model_configs import ModelOptimizations
from ..tt.tt_oftnet import TTOftNet
from ..tt.tt_encoder import TTObjectEncoder
from ..tt.tt_resnet import TTBasicBlock

from models.common.utility_functions import profiler

try:
    from tests.ttnn.unit_tests.base_functionality.test_bh_20_cores_sharding import skip_if_not_blackhole_20_cores
except ImportError:
    from tests.ttnn.unit_tests.test_bh_20_cores_sharding import skip_if_not_blackhole_20_cores


def frame_to_tensor(frame, pad_hw=(H_PADDED, W_PADDED), dtype=torch.float32):
    """Convert a BGR video frame to a preprocessed tensor."""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to float and normalize to [0, 1]
    frame_float = frame_rgb.astype(np.float32) / 255.0

    # Convert to tensor (C, H, W)
    tensor = torch.from_numpy(frame_float).permute(2, 0, 1)

    # Pad to required size
    pad_h, pad_w = pad_hw
    c, h, w = tensor.shape

    if h != pad_h or w != pad_w:
        padded = torch.zeros((c, pad_h, pad_w), dtype=dtype)
        padded[:, :h, :w] = tensor
        tensor = padded

    return tensor.to(dtype)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "video_path, calib_path, output_path",
    [
        (
            os.path.expanduser("~/work/oft_video/videos/0000.mp4"),  # Default video path
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources/000013.txt")),
            "output_video_ttnn.mp4",
        ),
    ],
)
@pytest.mark.parametrize(
    "model_dtype, oft_mode, frontend_mode",
    [
        (torch.float32, OftMode.OFT8, FrontendMode.REDUCED),
    ],
)
@torch.no_grad()
def test_video_inference(
    device,
    video_path,
    calib_path,
    output_path,
    model_dtype,
    model_location_generator,
    oft_mode,
    frontend_mode,
):
    skip_if_not_blackhole_20_cores(device)

    # Create output directory
    output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "."
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output will be saved to: {output_path}")

    # Load calibration
    logger.info(f"Loading calibration from {calib_path}...")
    calib = load_calib(calib_path, dtype=model_dtype)[None].to(model_dtype)

    # Create grid
    logger.info("Creating grid...")
    grid = make_grid(GRID_SIZE, (-GRID_SIZE[0] / 2.0, Y_OFFSET, 0.0), GRID_RES, dtype=model_dtype)[None].to(model_dtype)

    # Create reference model
    logger.info("Creating reference OFTNet model...")
    topdown_layers = 8
    ref_model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=topdown_layers,
        grid_res=GRID_RES,
        grid_height=GRID_HEIGHT,
        dtype=model_dtype,
        oft_mode=oft_mode,
        frontend_mode=frontend_mode,
    )

    # Load checkpoint
    logger.info("Loading checkpoint...")
    ref_model = load_checkpoint(ref_model, model_location_generator, oft_mode=oft_mode)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Fuse normalization
    done = fuse_imagenet_normalization(ref_model, ref_model.mean, ref_model.std)
    if not done:
        logger.warning("Normalization fusion not applied")

    # Create dummy input for parameter creation
    dummy_input = torch.zeros((1, 3, H_PADDED, W_PADDED), dtype=model_dtype)

    # Create state dict
    logger.info("Creating model parameters...")
    state_dict = create_OFT_model_parameters(ref_model, (dummy_input, calib, grid), device=device)
    state_dict = {"oftnet": state_dict}
    state_dict = fuse_conv_bn_parameters(state_dict)["oftnet"]

    # Apply optimizations
    model_opt = ModelOptimizations()
    model_opt.apply(state_dict)

    # Create reference encoder
    ref_encoder = ObjectEncoder(nms_thresh=NMS_THRESH, dtype=model_dtype)

    # Run model once on dummy input to get decoder parameters
    logger.info("Running model on dummy input to initialize decoder parameters...")
    with torch.no_grad():
        _, scores_init, pos_offsets_init, dim_offsets_init, ang_offsets_init = ref_model(dummy_input, calib, grid)

    # Squeeze to remove batch dimension as expected by decoder
    scores_init = scores_init.squeeze(0)
    pos_offsets_init = pos_offsets_init.squeeze(0)
    dim_offsets_init = dim_offsets_init.squeeze(0)
    ang_offsets_init = ang_offsets_init.squeeze(0)
    grid_ = grid.clone().squeeze(0)

    decoder_params = create_decoder_model_parameters(
        ref_encoder, [scores_init, pos_offsets_init, dim_offsets_init, ang_offsets_init, grid_], device
    )

    # Create TT models
    logger.info("Creating TT models...")
    tt_model = TTOftNet(
        device,
        state_dict,
        state_dict.layer_args,
        TTBasicBlock,
        [2, 2, 2, 2],
        ref_model.mean,
        ref_model.std,
        input_shape_hw=(H_PADDED, W_PADDED),
        calib=calib,
        grid=grid,
        topdown_layers=topdown_layers,
        grid_res=GRID_RES,
        grid_height=GRID_HEIGHT,
        oft_mode=oft_mode,
    )

    tt_encoder = TTObjectEncoder(device, decoder_params, grid_, nms_thresh=NMS_THRESH)

    # Prepare calibration and grid for TT
    tt_calib = ttnn.from_torch(calib, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_grid = ttnn.from_torch(grid, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    # Open video
    logger.info(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file {video_path}")
        pytest.skip(f"Could not open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    video_writer = None

    # Set up plot
    fig, ax = plt.subplots(nrows=1, figsize=(16, 9))

    profiler.start("video_processing")
    logger.info("Starting video processing...")
    frame_count = 0
    tt_input = None

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            profiler.start("frame_processing")
            # Convert frame to tensor
            input_tensor = frame_to_tensor(frame, dtype=model_dtype)[None]

            # Prepare TT input

            if tt_input is not None and tt_input.is_allocated():
                ttnn.deallocate(tt_input)
            tt_input = input_tensor.permute((0, 2, 3, 1))
            tt_input = ttnn.from_torch(tt_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

            # Preprocess input
            # tt_input = tt_model.preprocess(tt_input)

            # Run TT inference
            (_, _), tt_scores, tt_pos_offsets, tt_dim_offsets, tt_ang_offsets = tt_model.forward(
                device, tt_input, tt_calib, tt_grid
            )

            # Decode predictions
            tt_scores = ttnn.to_layout(ttnn.squeeze(tt_scores, 0), layout=ttnn.ROW_MAJOR_LAYOUT)
            tt_pos_offsets = ttnn.to_layout(ttnn.squeeze(tt_pos_offsets, 0), layout=ttnn.TILE_LAYOUT)
            tt_dim_offsets = ttnn.to_layout(ttnn.squeeze(tt_dim_offsets, 0), layout=ttnn.TILE_LAYOUT)
            tt_ang_offsets = ttnn.to_layout(ttnn.squeeze(tt_ang_offsets, 0), layout=ttnn.TILE_LAYOUT)
            tt_outs, _, _, _ = tt_encoder.decode(device, tt_scores, tt_pos_offsets, tt_dim_offsets, tt_ang_offsets)
            profiler.end("frame_processing")
            profiler.start("frame_postprocessing")
            tt_outs_torch = tt_encoder.decoder_postprocess(*tt_outs)
            ttnn.deallocate(tt_scores)
            ttnn.deallocate(tt_pos_offsets)
            ttnn.deallocate(tt_dim_offsets)
            ttnn.deallocate(tt_ang_offsets)
            for tt_out in tt_outs:
                ttnn.deallocate(tt_out)
            tt_objects = tt_encoder.create_objects(*tt_outs_torch)

            # Visualize
            ax.clear()
            input_tensor_viz = input_tensor.squeeze(0)
            visualize_objects(input_tensor_viz, calib, tt_objects, cmap="#00D9FF", ax=ax)
            ax.set_title(f"TTNN Detections - Frame {frame_count}")

            # Convert plot to image
            fig.canvas.draw()
            plot_img_buf = fig.canvas.buffer_rgba()
            plot_img_np = np.asarray(plot_img_buf)
            plot_img_bgr = cv2.cvtColor(plot_img_np, cv2.COLOR_RGBA2BGR)
            profiler.end("frame_postprocessing")

            profiler.start("video_writing")
            # Initialize video writer on first frame
            if video_writer is None:
                h, w, _ = plot_img_bgr.shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                logger.info(f"Output video resolution: {w}x{h} at {fps} FPS")

            video_writer.write(plot_img_bgr)
            profiler.end("video_writing")

            frame_count += 1
            if frame_count % 10 == 0:
                logger.info(f"Processed {frame_count} frames")

            # logger.warning("=== TTNN L1 Buffers ===")
            buffers = ttnn._ttnn.reports.get_buffers(device)
            l1_buffers = [buf for buf in buffers if buf.buffer_type == ttnn.BufferType.L1]
            for i, buf in enumerate(l1_buffers):
                logger.warning(
                    f"L1 Buffer {i}: addr={buf.address}, size={buf.max_size_per_bank}, layout={buf.buffer_layout}"
                )
            assert len(l1_buffers) == 0, "L1 memory leak detected!"

    finally:
        # Cleanup
        profiler.end("video_processing")
        logger.info(f"Done. Processed {frame_count} frames.")
        cap.release()
        if video_writer:
            video_writer.release()
            logger.info(f"Video saved to {output_path}")
        plt.close(fig)
        for key, data in profiler.times.items():
            logger.info(f"{key}: {data[-1]:.2f} seconds")
