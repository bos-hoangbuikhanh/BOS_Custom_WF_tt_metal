# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import math

import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


def get_core_grid_from_num_cores(num_cores: int, grid_rows: int, grid_cols: int):
    columns = num_cores // grid_rows
    assert columns <= grid_cols, "Not enough cores for specified core grid"
    ranges = []
    if columns != 0:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_rows - 1, columns - 1)))
    remainder = num_cores % grid_rows
    if remainder != 0:
        assert columns + 1 <= grid_cols, "Not enough cores for specified core grid"
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, columns), ttnn.CoreCoord(remainder - 1, columns)))
    return ttnn.CoreRangeSet({*ranges})


def find_closest_largest_divisor(num: int, start_divisor: int) -> int:
    divisor = start_divisor
    while num % divisor != 0:
        divisor -= 1
    return divisor


# Determins input memory config for a height sharded conv operation.
# If override_num_cores is set to True, the number of cores will be overriden to the closest largest divisor of the number of tiles
# This will avoid default conv codepath which can pad-up the nhw num tiles and produce padded output
# This can lead to issues with data-movment ops not handling padding correctly
def get_conv_input_memory_config(
    batch_size: int,
    input_channels: int,
    input_height: int,
    input_width: int,
    output_channels: int,
    output_height: int,
    output_width: int,
    compute_grid: ttnn.CoreGrid,
    input_channels_alignment: int,
    override_num_cores: bool,
) -> ttnn.MemoryConfig:
    if not isinstance(compute_grid, ttnn.CoreGrid):
        parallel_config_grid = ttnn.CoreGrid(x=compute_grid.x, y=compute_grid.y)
    else:
        parallel_config_grid = compute_grid

    if override_num_cores:
        nhw_ntiles = math.ceil(batch_size * output_height * output_width / 32)
        num_cores_nwh = find_closest_largest_divisor(nhw_ntiles, compute_grid.x * compute_grid.y)
        parallel_config_grid = get_core_grid_from_num_cores(num_cores_nwh, compute_grid.x, compute_grid.y)

    memory_config = ttnn.create_sharded_memory_config(
        shape=(
            math.ceil((input_width * input_height * batch_size) / (parallel_config_grid.x * parallel_config_grid.y)),
            math.ceil(input_channels / input_channels_alignment) * input_channels_alignment,
        ),
        core_grid=parallel_config_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return memory_config


def pcc_check(golden_torch, output_torch, batch_size, kind=""):
    _, pcc = check_with_pcc_without_tensor_printout(golden_torch, output_torch)
    pcc_batch = []
    for i in range(batch_size):
        _, pcc_batch_i = check_with_pcc_without_tensor_printout(golden_torch[i], output_torch[i])
        pcc_batch.append(pcc_batch_i)
    logger.debug(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> PCC({kind}) = {pcc}")
    for i in range(batch_size):
        logger.debug(
            f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> PCC({kind} / batch {i+1}) = {pcc_batch[i]}"
        )


def fold_like_ttnn(input_tensor, stride_h, stride_w, pad_c=0, pad_h=0, pad_w=0, output_shape=None):
    """
    Parameters:
        input_tensor: (N, C, H, W) shape torch.Tensor
        stride_h, stride_w: folding stride
        pad_c, pad_h, pad_w: C/H/W padding
        output_shape: (N, H_out, W_out, C_out) output shape

    Returns:
        (N, H_out, W_out, C_out) shape torch.Tensor
    """
    N, C, H, W = input_tensor.shape

    if pad_c > 0:
        pad_c_tensor = torch.zeros((N, pad_c, H, W), dtype=input_tensor.dtype, device=input_tensor.device)
        input_tensor = torch.cat([input_tensor, pad_c_tensor], dim=1)
        C = C + pad_c  # Update C

    input_tensor = torch.nn.functional.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h))  # (left, right, top, bottom)
    H += 2 * pad_h
    W += 2 * pad_w

    x = input_tensor.permute(0, 3, 1, 2)

    assert W % stride_w == 0, f"W ({W}) must be divisible by stride_w ({stride_w})"
    x = x.reshape(N, W // stride_w, C * stride_w, H)

    x = x.permute(0, 1, 3, 2)

    assert H % stride_h == 0, f"H ({H}) must be divisible by stride_h ({stride_h})"
    x = x.reshape(N, W // stride_w, H // stride_h, C * stride_w * stride_h)

    x = x.permute(0, 2, 1, 3)

    if output_shape is not None:
        n_out, h_out, w_out, c_out = output_shape
        x = x[:, :h_out, :w_out, :c_out]

    return x
