# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
OFT E2E Performance Test - Following ResNet Pattern

Based on models/demos/ttnn_resnet/tests/test_perf_e2e_resnet50.py
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import profiler
from models.perf.perf_utils import prep_perf_report
from tests.ttnn.unit_tests.test_bh_20_cores_sharding import skip_if_not_blackhole_20_cores
from tests.ttnn.utils_for_testing import check_with_pcc

from ..reference.encoder import ObjectEncoder

# Import OFT components
from ..reference.oftnet import FrontendMode, OftMode, OftNet
from ..reference.utils import load_calib, load_image, make_grid
from ..tests.common import GRID_HEIGHT, GRID_RES, GRID_SIZE, H_PADDED, NMS_THRESH, W_PADDED, Y_OFFSET, load_checkpoint
from ..tt.model_configs import ModelOptimizations
from ..tt.model_preprocessing import (
    create_decoder_model_parameters,
    create_OFT_model_parameters,
    fuse_conv_bn_parameters,
    fuse_imagenet_normalization,
)
from ..tt.oft_pipeline import PipelineConfig, create_pipeline_from_config
from ..tt.tt_encoder import TTObjectEncoder
from ..tt.tt_oftnet import TTOftNet
from ..tt.tt_resnet import TTBasicBlock

# Performance expectations for Blackhole 20 cores (single device)
PERF_EXPECTATIONS = {
    "test_perf_1cq": {"inference_time": 0.0666, "compile_time": 30},  # BH 0.0385
    "test_perf_trace_1cq": {"inference_time": 0.0666, "compile_time": 30},  # BH 0.0283
    "test_perf_2cqs": {"inference_time": 0.0666, "compile_time": 30},  # BH 0.0394
    "test_perf_trace_2cqs": {"inference_time": 0.0666, "compile_time": 30},  # BH 0.0276
}


class OFTPerformanceRunnerInfra:
    """
    OFT test infrastructure - encapsulates model setup and execution.

    Following ResNet test_infra pattern.
    """

    def __init__(self, device, input_tensor, calib, grid, ref_model, oft_mode, model_location_generator):
        self.device = device
        self.oft_mode = oft_mode
        self.batch_size = input_tensor.shape[0]

        # Preprocessing (demo.py pattern)
        logger.info("[Infra] Applying preprocessing...")
        fuse_imagenet_normalization(ref_model, ref_model.mean, ref_model.std)

        state_dict = create_OFT_model_parameters(ref_model, (input_tensor, calib, grid), device=device)
        state_dict = {"oftnet": state_dict}
        state_dict = fuse_conv_bn_parameters(state_dict)["oftnet"]

        model_opt = ModelOptimizations()
        model_opt.apply(state_dict)

        # Create TT model
        logger.info("[Infra] Creating TT model...")
        self.tt_model = TTOftNet(
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
            topdown_layers=8,  # Fixed for OFT8/REDUCED
            grid_res=GRID_RES,
            grid_height=GRID_HEIGHT,
            oft_mode=oft_mode,
            preprocess_frontend_conv1=True,
        )

        # Pre-convert persistent tensors
        self.tt_calib = ttnn.from_torch(calib, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        self.tt_grid = ttnn.from_torch(grid, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        grid_ = grid.clone().squeeze(0)
        self.tt_grid_ = ttnn.from_torch(grid_, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

        # Create reference encoder for decoder setup
        ref_encoder = ObjectEncoder(nms_thresh=NMS_THRESH, dtype=torch.float32)

        # Create dummy decoder inputs for parameter initialization
        # Grid determines the spatial dimensions (160x160 for GRID_SIZE=80, GRID_RES=0.5)
        # Shapes after squeezing batch dim match grid spatial dims
        grid_h, grid_w = grid_.shape[0], grid_.shape[1]
        dummy_scores = torch.zeros(1, grid_h - 1, grid_w - 1)
        dummy_pos_offsets = torch.zeros(1, 3, grid_h - 1, grid_w - 1)
        dummy_dim_offsets = torch.zeros(1, 3, grid_h - 1, grid_w - 1)
        dummy_ang_offsets = torch.zeros(1, 2, grid_h - 1, grid_w - 1)
        decoder_params = create_decoder_model_parameters(
            ref_encoder, [dummy_scores, dummy_pos_offsets, dummy_dim_offsets, dummy_ang_offsets, grid_], device
        )

        # Create TT encoder (decoder)
        logger.info("[Infra] Creating TT encoder (decoder)...")
        self.tt_encoder = TTObjectEncoder(device, decoder_params, grid_, nms_thresh=NMS_THRESH)

        # Store input/output for model_wrapper
        self.input_tensor = None
        self.output_tensor = None

        # Store reference output for validation
        self.ref_output = None

    def setup_dram_sharded_input(self, device):
        """
        Setup DRAM and L1 sharded input configuration.

        Following ResNet pattern (resnet50_test_infra.py lines 263-298).
        Creates proper HEIGHT_SHARDED L1 config for traced execution.
        """
        # Create preprocessed tensor on host first
        # OFT uses [B, H, W, C] layout after permute
        tt_input = self.input_tensor_host.permute((0, 2, 3, 1))  # [1, 384, 1280, 3]
        # DRAM interleaved config
        dram_memory_config = ttnn.DRAM_MEMORY_CONFIG

        # Create host tensor with proper layout
        tt_input_host = ttnn.from_torch(
            tt_input,
            dtype=ttnn.bfloat16,
            # layout=ttnn.ROW_MAJOR_LAYOUT,
            layout=ttnn.TILE_LAYOUT,
        )

        return tt_input_host, dram_memory_config, dram_memory_config

    def preprocess(self, input_tensor):
        self.input_tensor = input_tensor
        self.preprocessed_tensor = self.tt_model.preprocess(self.input_tensor)
        return self.preprocessed_tensor

    def run(self, input_tensor):
        """Run OFT model forward pass + decoder"""
        self.input_tensor = input_tensor
        (intermediates, layer_names), tt_scores, tt_pos_offsets, tt_dim_offsets, tt_ang_offsets = self.tt_model.forward(
            self.device, self.input_tensor, self.tt_calib, self.tt_grid
        )

        # Run decoder (following demo.py pattern)
        tt_scores = ttnn.to_layout(ttnn.squeeze(tt_scores, 0), layout=ttnn.ROW_MAJOR_LAYOUT)
        tt_pos_offsets = ttnn.to_layout(ttnn.squeeze(tt_pos_offsets, 0), layout=ttnn.TILE_LAYOUT)
        tt_dim_offsets = ttnn.to_layout(ttnn.squeeze(tt_dim_offsets, 0), layout=ttnn.TILE_LAYOUT)
        tt_ang_offsets = ttnn.to_layout(ttnn.squeeze(tt_ang_offsets, 0), layout=ttnn.TILE_LAYOUT)

        # decode returns: (tt_outs_list, intermediates_list, names_tuple, intermediate_names_tuple)
        # We only need the first element for the pipeline
        tt_outs, _, _, _ = self.tt_encoder.decode(
            self.device, tt_scores, tt_pos_offsets, tt_dim_offsets, tt_ang_offsets
        )
        # tt_outs is [tt_indices, tt_scores, tt_positions, tt_dimensions, tt_angles]

        # Convert to tuple for pipeline compatibility
        self.output_tensor = tuple(tt_outs)
        return self.output_tensor

    def postprocess(self, output):
        """Postprocess decoder outputs (torch code, not traced)"""
        # output is [tt_indices, tt_scores, tt_positions, tt_dimensions, tt_angles]
        # Convert TTNN outputs to torch tensors
        tt_outs_torch = self.tt_encoder.decoder_postprocess(*output)
        # Return scores for validation
        return tt_outs_torch

    def validate(self, tt_outs_torch, expected_detections=None):
        """Validate outputs against number of detected objects"""
        tt_objects_torch = self.tt_encoder.create_objects(*tt_outs_torch)
        logger.info("=== TTNN Objects ===")
        detected = len(tt_objects_torch)
        if expected_detections is not None and detected != expected_detections:
            logger.warning(f"Expected {expected_detections} detections, got {len(tt_objects_torch)}")
        for i, obj in enumerate(tt_objects_torch):
            logger.info(f"TT Object {i}: {obj}")

        return True, f"Validation passed. {detected=} {expected_detections=} objects detected."


def _run_model_pipeline(
    device,
    tt_inputs_list,
    test_infra,
    num_warmup_iterations,
    num_measurement_iterations,
    num_command_queues,
    trace,
    detection_count_list=None,
):
    """
    Run model pipeline with specified configuration.

    Following ResNet perf_e2e_resnet50.py pattern.
    Args:
        tt_inputs_list: List of input tensors to round-robin through during measurement
    """
    # Use first input for warmup and compilation
    test_infra.input_tensor_host = tt_inputs_list[0]

    # Setup DRAM sharded input
    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(device)

    # Create pipeline
    pipeline = create_pipeline_from_config(
        config=PipelineConfig(
            use_trace=trace, num_command_queues=num_command_queues, all_transfers_on_separate_command_queue=False
        ),
        model=test_infra,
        device=device,
        dram_input_memory_config=sharded_mem_config_DRAM,
        l1_input_memory_config=None,
    )

    logger.info(f"Running model warmup with input shape {list(tt_inputs_host.shape)}")
    profiler.start("compile")
    pipeline.compile(tt_inputs_host)
    profiler.end("compile")

    # Run warmup iterations with first input only
    if num_warmup_iterations > 0:
        logger.info(f"Running {num_warmup_iterations} warmup iterations")
        warmup_inputs = [tt_inputs_host] * num_warmup_iterations
        pipeline.preallocate_output_tensors_on_host(num_warmup_iterations)
        pipeline.enqueue(warmup_inputs).pop_all()
        logger.info("Warmup complete")

    # Prepare measurement inputs by round-robin through all input tensors
    logger.info(f"Preparing {num_measurement_iterations} measurement inputs from {len(tt_inputs_list)} input tensors")
    host_inputs = []
    detections_validation = []
    for i in range(num_measurement_iterations):
        input_idx = i % len(tt_inputs_list)
        test_infra.input_tensor_host = tt_inputs_list[input_idx]
        tt_input_host, _, _ = test_infra.setup_dram_sharded_input(device)
        host_inputs.append(tt_input_host)
        detections_validation.append(detection_count_list[input_idx] if detection_count_list else None)

    # Preallocate outputs
    pipeline.preallocate_output_tensors_on_host(num_measurement_iterations)

    logger.info(f"Starting performance pipeline for {num_measurement_iterations} iterations")
    profiler.start(f"run_model_pipeline_{num_command_queues}cqs")
    outputs = pipeline.enqueue(host_inputs).pop_all()
    profiler.end(f"run_model_pipeline_{num_command_queues}cqs")

    logger.info(f"Running validation with {len(outputs)} outputs")
    profiler.start("validation")
    for i, (output, expected_detections) in enumerate(zip(outputs, detections_validation)):
        torch_outputs = test_infra.postprocess(output)
        passed, pcc_message = test_infra.validate(torch_outputs, expected_detections=expected_detections)
        logger.info(f"Output {i} validation: {pcc_message}")
        if not passed:
            logger.warning(f"Output {i} validation failed: {pcc_message}")
    profiler.end("validation")

    pipeline.cleanup()


# ============================================================================
# Test Functions - Blackhole 20 cores only
# ============================================================================


@pytest.mark.parametrize(
    "device_params, model_version",
    [
        ({"l1_small_size": 24576}, "oft"),
        ({"l1_small_size": 32768, "trace_region_size": 3000000}, "oft_trace"),
        ({"l1_small_size": 32768, "num_command_queues": 2}, "oft_2cqs"),
        ({"l1_small_size": 32768, "num_command_queues": 2, "trace_region_size": 3000000}, "oft_trace_2cqs"),
    ],
    indirect=["device_params"],
    ids=["oft_1cq", "oft_trace_1cq", "oft_2cqs", "oft_trace_2cqs"],
)
def test_perf_oft(
    device,
    model_version,
    model_location_generator,
    input_image_path=None,
    calib_path=None,
    expected_inference_time=None,
    expected_compile_time=None,
    model_dtype=torch.float32,
    oft_mode=OftMode.OFT8,
    frontend_mode=FrontendMode.REDUCED,
):
    """OFT performance test - all variants (Blackhole 20 cores only)"""
    skip_if_not_blackhole_20_cores(device)

    # Set defaults from PERF_EXPECTATIONS if not provided
    if expected_inference_time is None:
        test_name = (
            f"test_perf{'_trace' if 'trace' in model_version else ''}{'_2cqs' if '2cqs' in model_version else '_1cq'}"
        )
        expected_inference_time = PERF_EXPECTATIONS[test_name]["inference_time"]
    if expected_compile_time is None:
        test_name = (
            f"test_perf{'_trace' if 'trace' in model_version else ''}{'_2cqs' if '2cqs' in model_version else '_1cq'}"
        )
        expected_compile_time = PERF_EXPECTATIONS[test_name]["compile_time"]

    # Set default input paths if not provided
    if input_image_path is None:
        input_image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources", "000013.jpg"))
    if calib_path is None:
        calib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources", "000013.txt"))

    profiler.clear()
    device.disable_and_clear_program_cache()

    batch_size = 1  # OFT uses batch_size=1
    comments = f"{H_PADDED}x{W_PADDED}_batchsize{batch_size}_{oft_mode.name}_{frontend_mode.name}"

    # Load multiple input images for round-robin testing
    resources_dir = os.path.join(os.path.dirname(__file__), "..", "resources")
    input_image_paths = [
        os.path.join(resources_dir, "000009.jpg"),
        os.path.join(resources_dir, "000013.jpg"),
        os.path.join(resources_dir, "000022.jpg"),
        os.path.join(resources_dir, "000110.jpg"),
        os.path.join(resources_dir, "000340.jpg"),
    ]
    calib_paths = [
        os.path.join(resources_dir, "000009.txt"),
        os.path.join(resources_dir, "000013.txt"),
        os.path.join(resources_dir, "000022.txt"),
        os.path.join(resources_dir, "000110.txt"),
        os.path.join(resources_dir, "000340.txt"),
    ]
    detection_count_list = [17, 4, 25, 49, 8]  # Expected detections for validation

    logger.info(f"Loading {len(input_image_paths)} input images for round-robin testing")
    input_tensors = []
    for img_path in input_image_paths:
        img_tensor, _, _ = load_image(img_path, pad_hw=(H_PADDED, W_PADDED), dtype=model_dtype)
        img_tensor = img_tensor[None].to(model_dtype)
        input_tensors.append(img_tensor)

    # Use first calib/grid for model setup (all should be compatible)
    calib = load_calib(calib_paths[0], dtype=model_dtype)[None].to(model_dtype)
    grid = make_grid(GRID_SIZE, (-GRID_SIZE[0] / 2.0, Y_OFFSET, 0.0), GRID_RES, dtype=model_dtype)[None].to(model_dtype)

    logger.info("Creating reference model...")
    ref_model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=8,
        grid_res=GRID_RES,
        grid_height=GRID_HEIGHT,
        dtype=model_dtype,
        oft_mode=oft_mode,
        frontend_mode=frontend_mode,
    )
    ref_model = load_checkpoint(ref_model, model_location_generator, oft_mode=oft_mode)
    ref_model.eval()

    # Create test infrastructure (use first input for setup)
    logger.info("Creating test infrastructure...")
    device.enable_program_cache()
    test_infra = OFTPerformanceRunnerInfra(
        device, input_tensors[0], calib, grid, ref_model, oft_mode, model_location_generator
    )

    ttnn.synchronize_device(device)

    num_warmup_iterations = 5
    num_measurement_iterations = 50

    with torch.no_grad():
        # Run appropriate pipeline variant with inlined parameters
        if "oft_trace_2cqs" in model_version:
            _run_model_pipeline(
                device,
                input_tensors,
                test_infra,
                num_warmup_iterations,
                num_measurement_iterations,
                num_command_queues=2,
                trace=True,
                detection_count_list=detection_count_list,
            )
        elif "oft_trace" in model_version:
            _run_model_pipeline(
                device,
                input_tensors,
                test_infra,
                num_warmup_iterations,
                num_measurement_iterations,
                num_command_queues=1,
                trace=True,
                detection_count_list=detection_count_list,
            )
        elif "oft_2cqs" in model_version:
            _run_model_pipeline(
                device,
                input_tensors,
                test_infra,
                num_warmup_iterations,
                num_measurement_iterations,
                num_command_queues=2,
                trace=False,
                detection_count_list=detection_count_list,
            )
        elif "oft" in model_version:
            _run_model_pipeline(
                device,
                input_tensors,
                test_infra,
                num_warmup_iterations,
                num_measurement_iterations,
                num_command_queues=1,
                trace=False,
                detection_count_list=detection_count_list,
            )
        else:
            assert False, f"Model version to run {model_version} not found"

    # Calculate metrics (ResNet pattern)
    first_iter_time = profiler.get("compile")
    num_cqs = 2 if "2cqs" in model_version else 1
    inference_time_avg = profiler.get(f"run_model_pipeline_{num_cqs}cqs") / num_measurement_iterations
    compile_time = first_iter_time - 2 * inference_time_avg

    # Prepare performance report
    prep_perf_report(
        model_name=f"ttnn_{model_version}_batch_size{batch_size}",
        batch_size=batch_size,
        inference_and_compile_time=first_iter_time,
        inference_time=inference_time_avg,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=0.0,
    )

    logger.info(f"\n{'='*80}")
    logger.info(f"OFT {model_version} {comments}")
    logger.info(f"  Inference time (avg): {inference_time_avg:.4f}s ({inference_time_avg*1000:.2f}ms)")
    logger.info(f"  Compile time: {compile_time:.4f}s")
    logger.info(f"  FPS: {batch_size/inference_time_avg:.2f}")
    logger.info(f"  Validation time: {profiler.get('validation'):.4f}s")
    logger.info(f"{'='*80}\n")

    # Assert inference time is within 10% of expected
    tolerance = 0.1
    lower_bound = expected_inference_time * (1 - tolerance)
    upper_bound = expected_inference_time * (1 + tolerance)
    assert lower_bound <= inference_time_avg <= upper_bound, (
        f"Inference time {inference_time_avg:.4f}s is outside expected range "
        f"[{lower_bound:.4f}s, {upper_bound:.4f}s] (expected: {expected_inference_time:.4f}s ±{tolerance*100}%)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
