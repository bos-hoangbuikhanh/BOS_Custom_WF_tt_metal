# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
from loguru import logger

import ttnn
from models.bos_model.yolov10x.common import YOLOV10_L1_SMALL_SIZE
from models.bos_model.yolov10x.runner.performant_runner_infra import YOLOv10PerformanceRunnerInfra
from models.bos_model.yolov10x.runner.pipeline_runner import YoloV10xPipelineRunner
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config


def run_yolov10x_inference(
    device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
):
    inputs_mesh_mapper, weights_mesh_mapper, outputs_mesh_composer = get_mesh_mappers(device)

    test_infra = YOLOv10PerformanceRunnerInfra(
        device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        model_location_generator,
        resolution=resolution,
        use_pretrained_weight=True,
        mesh_mapper=inputs_mesh_mapper,
        weights_mesh_mapper=weights_mesh_mapper,
        mesh_composer=outputs_mesh_composer,
    )

    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(device)

    pipeline = create_pipeline_from_config(
        config=PipelineConfig(use_trace=True, num_command_queues=2, all_transfers_on_separate_command_queue=False),
        model=YoloV10xPipelineRunner(test_infra),
        device=device,
        dram_input_memory_config=sharded_mem_config_DRAM,
        l1_input_memory_config=input_mem_config,
    )

    logger.info(f"Running model warmup with input shape {list(tt_inputs_host.shape)}")
    pipeline.compile(tt_inputs_host)

    batch_size = 1

    # Run inference
    t0 = time.time()
    for _ in range(10):
        _ = pipeline.enqueue([tt_inputs_host]).pop_all()

    t1 = time.time()

    inference_time_avg = round((t1 - t0) / 10, 6)
    logger.info(
        f"ttnn_yolov10_batch_size: {batch_size}, resolution: {resolution}. One inference iteration time (sec): {inference_time_avg}, FPS: {round( batch_size / inference_time_avg)}"
    )

    pipeline.cleanup()


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV10_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat8_b),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (224, 224),
        (320, 320),
    ],
    ids=["res224", "res320"],
)
def test_e2e_performant(
    device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
):
    run_yolov10x_inference(
        device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        model_location_generator,
        resolution,
    )
