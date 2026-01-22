# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report


def run_device_perf_tests(
    command, expected_device_perf_cycles_per_iteration, subdir, model_name, num_iterations=1, batch_size=1, margin=0.015
):
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL DURATION [ns]"
    post_processed_results = run_device_perf(
        command, subdir=subdir, num_iterations=num_iterations, cols=cols, batch_size=batch_size
    )
    expected_perf_cols = {inference_time_key: expected_device_perf_cycles_per_iteration}
    expected_results = check_device_perf(
        post_processed_results, margin=margin, expected_perf_cols=expected_perf_cols, assert_on_fail=True
    )
    prep_device_perf_report(
        model_name=model_name,
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )



@pytest.mark.parametrize(
    "command, expected_device_perf_cycles_per_iteration, subdir, model_name, num_iterations, batch_size, margin",
    [
        (
            f"pytest {os.path.dirname(__file__)}/pcc/test_oftnet.py::test_oftnet -k fp32_oft8",
            27_999_174,
            "oft_oftnet",
            "oft_oftnet",
            1,
            1,
            0.015,
        ),
        (
            f"pytest {os.path.dirname(__file__)}/pcc/test_encoder.py::test_decode",
            4_569_515,
            "oft_decoder",
            "oft_decoder",
            1,
            1,
            0.015,
        ),
        (
            f"pytest {os.path.abspath(os.path.join(os.path.dirname(__file__), '../demo/demo.py'))}::test_demo_inference -k OFT8",
            30_743_019,
            "oft_full_oft8",
            "oft_full_oft8",
            1,
            1,
            0.015,
        ),
    ],
    ids=[
        "device_perf_oft_oftnet",
        "device_perf_oft_decoder",
        "device_perf_oft_full_demo_oft8",
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_device_perf_oft(
    command, expected_device_perf_cycles_per_iteration, subdir, model_name, num_iterations, batch_size, margin
):
    from loguru import logger
    logger.warning(f"{command=}")
    run_device_perf_tests(
        command,
        expected_device_perf_cycles_per_iteration,
        subdir=subdir,
        model_name=model_name,
        num_iterations=num_iterations,
        batch_size=batch_size,
        margin=margin,
    )