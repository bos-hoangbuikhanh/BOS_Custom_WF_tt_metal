# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.perf.device_perf_utils import run_model_device_perf_test


@pytest.mark.parametrize(
    "command, expected_device_perf_ns_per_iteration, subdir, model_name, num_iterations, batch_size, margin, comments",
    [
        (
            "pytest models/bos_model/oft/tests/pcc/test_oftnet.py::test_oftnet -k fp32_use_device_oft",
            396_744_966,  # BH-bfp16: 195_418_371, A0-fp32: 396_744_966
            "oft_oftnet",
            "oft_oftnet",
            1,
            1,
            0.015,
            "",
        ),
        (
            "pytest models/bos_model/oft/tests/pcc/test_encoder.py::test_decode",
            8_410_909,  # BH: 4_145_454, A0: 8_410_909
            "oft_decoder",
            "oft_decoder",
            1,
            1,
            0.015,
            "",
        ),
        (
            "pytest models/bos_model/oft/demo/demo.py::test_demo_inference",
            401_314_620,  # BH: 198_042_833, A0: 401_314_620
            "oft_full_demo",
            "oft_full_demo",
            1,
            1,
            0.015,
            "",
        ),
    ],
    ids=[
        "device_perf_oft_oftnet",
        "device_perf_oft_decoder",
        "device_perf_oft_full",
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_device_perf_oft(
    command, expected_device_perf_ns_per_iteration, subdir, model_name, num_iterations, batch_size, margin, comments
):
    run_model_device_perf_test(
        command=command,
        expected_device_perf_ns_per_iteration=expected_device_perf_ns_per_iteration,
        subdir=subdir,
        model_name=model_name,
        num_iterations=num_iterations,
        batch_size=batch_size,
        margin=margin,
        comments=comments,
    )
