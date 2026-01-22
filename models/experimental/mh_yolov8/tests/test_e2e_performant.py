# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import torch
from loguru import logger
import ttnn
from models.bos_model.mh_yolov8.tests.yolov8s_e2e_performant import Yolov8sTrace2CQ, Yolov8sTrace
from models.common.utility_functions import run_for_wormhole_b0


# @run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "input_shape",
    ((1, 16, 320, 320),),
)
def test_run_yolov8s_trace_2cqs_inference(
    device,
    use_program_cache,
    input_shape,
    model_location_generator,
):
    batch_size, input_channel, inp_h, inp_w = input_shape
    yolov8s_trace_2cq = Yolov8sTrace2CQ()

    yolov8s_trace_2cq.initialize_yolov8s_trace_2cqs_inference(
        device,
        batch_size,
        (inp_h, inp_w),
    )

    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
    torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1).reshape(
        1, 1, batch_size * inp_h * inp_w, -1
    )  # need for from_torch
    # tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    inference_iter_count = 10
    inference_time_iter = []
    for iter in range(0, inference_iter_count):
        t0 = time.time()
        tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        # tt_inputs_host, _, _ = yolov8s_trace_2cq.test_infra.setup_dram_sharded_input(device, torch_input_tensor)
        outputs = yolov8s_trace_2cq.execute_yolov8s_trace_2cqs_inference(tt_inputs_host)
        output0 = ttnn.from_device(outputs[0], blocking=True)
        t1 = time.time()
        inference_time_iter.append(t1 - t0)
    yolov8s_trace_2cq.release_yolov8s_trace_2cqs_inference()
    inference_time_avg = round(sum(inference_time_iter) / len(inference_time_iter), 6)
    logger.info(
        f"ttnn_yolov8s_320x320_batch_size_{batch_size}. One inference iteration time (sec): {inference_time_avg}, FPS: {round(batch_size/inference_time_avg)}"
    )


# @run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 6434816}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",
    ((1, 16, 320, 320),),
)
def test_run_yolov8s_trace_inference(
    device,
    use_program_cache,
    input_shape,
    model_location_generator,
):
    batch_size, input_channel, inp_h, inp_w = input_shape
    yolov8s_trace = Yolov8sTrace()

    yolov8s_trace.initialize_yolov8s_trace_inference(
        device,
        batch_size,
        (inp_h, inp_w),
    )

    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
    torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1).reshape(
        1, 1, batch_size * inp_h * inp_w, -1
    )  # need for from_torch
    # tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    inference_iter_count = 10
    inference_time_iter = []
    for iter in range(0, inference_iter_count):
        t0 = time.time()
        tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        # tt_inputs_host, _ = yolov8s_trace.test_infra.setup_l1_sharded_input(device, torch_input_tensor)
        outputs = yolov8s_trace.execute_yolov8s_trace_inference(tt_inputs_host)
        output0 = ttnn.from_device(outputs[0], blocking=True)
        t1 = time.time()
        inference_time_iter.append(t1 - t0)
    yolov8s_trace.release_yolov8s_trace_inference()
    inference_time_avg = round(sum(inference_time_iter) / len(inference_time_iter), 6)
    logger.info(
        f"ttnn_yolov8s_320x320_batch_size_{batch_size}. One inference iteration time (sec): {inference_time_avg}, FPS: {round(batch_size/inference_time_avg)}"
    )
