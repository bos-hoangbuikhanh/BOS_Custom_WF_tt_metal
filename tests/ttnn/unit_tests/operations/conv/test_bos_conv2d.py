# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from tests.ttnn.nightly.unit_tests.operations.conv.test_conv2d import run_conv, HS, WS
import ttnn

SliceHeight = ttnn.Conv2dSliceHeight
SliceWidth = ttnn.Conv2dSliceWidth


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width, shard_layout, config, filter, stride, padding",
    (
        # (16, 16, 180, 540, HS, {"act_block_h": 32}, 7, 2, 3),  # 0.999 pcc # A0 passed
        # (64, 64, 45, 135, HS, {"act_block_h": 32}, 3, 1, 1),  # 0.999 # A0 passed
        # (128, 64, 45, 135, HS, {"act_block_h": 32}, 3, 2, 1),  # 0.999 # A0 passed
        # (128, 128, 23, 68, HS, {"act_block_h": 32}, 3, 1, 1),  # 0.999 # A0 passed
        # (128, 64, 45, 135, HS, {"act_block_h": 32}, 1, 2, 0),  # 0.999 # A0 passed
        # (256, 128, 23, 68, HS, {"act_block_h": 32}, 3, 2, 1),  # 0.999 # A0 passed
        (256, 256, 12, 34, HS, {"act_block_h": 32}, 3, 1, 1),  # 0.999 # A0 passed
        (256, 256, 12, 34, HS, {"act_block_h": 32}, 3, 1, 1),  # 0.999 # A0 passed
        # (256, 128, 23, 68, HS, {"act_block_h": 32}, 1, 2, 0),  # 0.999 # A0 passed
        # (512, 256, 12, 34, HS, {"act_block_h": 32}, 3, 2, 1),  # 0.999 # A0 passed
        # (512, 512, 6, 17, None, None, 3, 1, 1),  # 0.999 # A0 passed
        # (512, 256, 12, 34, HS, {"act_block_h": 32}, 1, 2, 0),  # 0.999 # A0 passed
        # (256, 128, 23, 68, HS, {"act_block_h": 32}, 1, 1, 0),  # 0.999 # A0 passed
        # (256, 256, 12, 34, HS, {"act_block_h": 32}, 1, 1, 0),  # 0.999 # A0 passed
        # (256, 512, 6, 17, HS, {"act_block_h": 32}, 1, 1, 0),  # 0.999 # A0 passed
        # (256, 256, 159, 159, HS, {"act_block_h": 32}, 3, 1, 1), # OOM: use conv_dram
        # (9, 256, 159, 159, HS, {"act_block_h": 64}, 3, 1, 1), # OOM: use conv_dram
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "input_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "fp32_accum",
    [False],
)
@pytest.mark.parametrize(
    "packer_l1_acc",
    [False],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
def test_conv_oft(
    device,
    # torch_tensor_map,
    math_fidelity,
    output_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    shard_layout,
    config,
    filter,
    stride,
    padding,
    output_layout,
    fp32_accum,
    packer_l1_acc,
    input_dtype,
):
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and shard_layout == WS:
        pytest.skip("Bug in Width Sharded Row Major Tensor Creation when height%32!=0. #19408")

    if output_layout == ttnn.ROW_MAJOR_LAYOUT and output_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")

    if output_layout == ttnn.ROW_MAJOR_LAYOUT and output_dtype == ttnn.bfloat16 and packer_l1_acc and fp32_accum:
        pytest.skip("skipping due to pack_untilize_dst issue!")

    torch_tensor_map = {}
    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        output_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter,
        filter,
        stride,
        stride,
        padding,
        config,
        shard_layout=shard_layout,
        output_layout=output_layout,
        has_bias=True,
        fp32_accum=fp32_accum,
        packer_l1_acc=packer_l1_acc,
        run_twice=True,
        input_layout=ttnn.TILE_LAYOUT if input_dtype == ttnn.bfloat8_b else None,
        input_dtype=input_dtype,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_channels, output_channels, input_height, input_width, slice_type, num_slices, weights_dtype, output_dtype, kernel, stride, padding, dilation, act_block_h_override,  math_fidelity",
    (
        (
            1,
            256,
            256,
            159,
            159,
            SliceHeight,
            4,
            ttnn.bfloat16,
            ttnn.bfloat16,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            0,
            ttnn.MathFidelity.LoFi,
        ),  # 0.999
        (
            1,
            256,
            9,
            159,
            159,
            SliceHeight,
            4,
            ttnn.bfloat16,
            ttnn.bfloat16,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            0,
            ttnn.MathFidelity.LoFi,
        ),  # 0.999
    ),
)
@pytest.mark.parametrize(
    "has_bias, fp32_accum, packer_l1_acc",
    [[False, False, False]],
)
@pytest.mark.parametrize(
    "input_layout",
    [ttnn.ROW_MAJOR_LAYOUT],
)
def test_conv_dram_oft(
    device,
    # torch_tensor_map,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    has_bias,
    weights_dtype,
    output_dtype,
    slice_type,
    num_slices,
    kernel,
    stride,
    padding,
    dilation,
    act_block_h_override,
    math_fidelity,
    fp32_accum,
    input_layout,
    packer_l1_acc,
):
    if device.core_grid.y == 7:
        pytest.skip("Tests have been configured for N150.")
    config = {
        "act_block_h": act_block_h_override,
    }
    torch_tensor_map = {}
    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        output_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        kernel[0],
        kernel[1],
        stride[0],
        stride[1],
        padding,
        config,
        has_bias=True,
        fp32_accum=fp32_accum,
        packer_l1_acc=packer_l1_acc,
        input_layout=input_layout,
        run_twice=False,
        fast_compare=False,
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=slice_type,
            num_slices=num_slices,
        ),
    )
