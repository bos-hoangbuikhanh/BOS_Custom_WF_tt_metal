import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# Error:
# E       RuntimeError: TT_FATAL @ /home/bos/work/shivam/tt-metal/ttnn/cpp/ttnn/operations/normalization/groupnorm/groupnorm.cpp:69: (nhw % ttnn::types::TILE_SIZE) == 0
# E       info:
# E       Invalid tensor dimensions: The product of NHW dimensions (468) must be divisible by the tile size (32)


@pytest.mark.parametrize(
    "N, C, H, W, num_groups",
    [
        # (1, 64, 192, 624, 16),
        # (1, 64, 96, 312, 16),
        # (1, 128, 48, 156, 16),
        # (1, 256, 24, 78, 16),
        # (1, 512, 12, 39, 16),
        # (1, 256, 48, 156, 16),
        # (1, 256, 12, 39, 16),
        # (1, 256, 159, 159, 16),
    ],
)
def test_group_norm_with_height_sharded_oft(device, N, C, H, W, num_groups):
    torch.manual_seed(0)

    grid_size = ttnn.CoreGrid(y=1, x=4)

    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias
    )
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_tensor = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # input mask
    input_mask_tensor = ttnn.create_group_norm_input_mask(C, num_groups, grid_size.y)
    input_mask_tensor = ttnn.from_torch(
        input_mask_tensor,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    gamma = ttnn.create_group_norm_weight_bias_rm(torch_weight, C, grid_size.y)
    beta = ttnn.create_group_norm_weight_bias_rm(torch_bias, C, grid_size.y)

    gamma_t = ttnn.from_torch(
        gamma,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_t = ttnn.from_torch(
        beta,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.group_norm(
        input_tensor,
        num_groups=num_groups,
        input_mask=input_mask_tensor,
        weight=gamma_t,
        bias=beta_t,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        core_grid=grid_size,
    )

    output_tensor = ttnn.to_memory_config(output_tensor, ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize(
    "N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x",
    [
        # For res 180*540
        (1, 64, 90, 270, 16, 2, 4, 1),  # Static Buffer
        (1, 64, 45, 135, 16, 2, 2, 1),  # 0.999
        (1, 128, 23, 68, 16, 2, 4, 1),  # 0.999
        (1, 256, 12, 34, 16, 2, 4, 1),  # 0.999
        (1, 512, 6, 17, 16, 2, 2, 1),  # 0.986
        (1, 256, 23, 68, 16, 2, 4, 1),  # 0.999
        (1, 256, 6, 17, 16, 2, 4, 1),  # 0.986
        (1, 256, 159, 159, 16, 2, 4, 1),  # Static Buffer
        # For res 384*1248
        (1, 64, 192, 624, 16, 2, 4, 1),  # Static Buffer
        (1, 64, 96, 312, 16, 2, 4, 1),  # Static Buffer
        (1, 128, 48, 156, 16, 2, 4, 1),  # Static Buffer
        (1, 256, 24, 78, 16, 2, 4, 1),  # 0.999
        (1, 512, 12, 39, 16, 2, 2, 1),  # 0.999
        (1, 256, 48, 156, 16, 2, 4, 1),  # Static Buffer
        (1, 256, 12, 39, 16, 2, 4, 1),  # 0.999
        (1, 256, 159, 159, 16, 2, 4, 1),  # Static Buffer
    ],
)
def test_group_norm_DRAM_oft(device, N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip()

    grid_size = ttnn.CoreGrid(y=cores_y, x=cores_x)

    # torch input tensor
    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias
    )

    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    # input tensor
    input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_tensor_row_major = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_tilized = ttnn.tilize_with_zero_padding(input_tensor_row_major, use_multicore=True)

    # input mask
    input_mask_tensor = ttnn.create_group_norm_input_mask(C, num_groups, grid_size.y)
    input_mask_tensor = ttnn.from_torch(
        input_mask_tensor,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # gamma/beta
    gamma = ttnn.create_group_norm_weight_bias_rm(torch_weight, C, grid_size.y)
    beta = ttnn.create_group_norm_weight_bias_rm(torch_bias, C, grid_size.y)

    gamma_t = ttnn.from_torch(
        gamma,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_t = ttnn.from_torch(
        beta,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # groupnorm
    output_tensor = ttnn.group_norm(
        input_tensor_tilized,
        num_groups=num_groups,
        input_mask=input_mask_tensor,
        weight=gamma_t,
        bias=beta_t,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_layout=ttnn.TILE_LAYOUT,
        core_grid=grid_size,
        inplace=False,
        num_out_blocks=num_out_blocks,
    )

    # output tensor
    print(f"Tt Output tensor: {output_tensor.shape}")
    print(f"Torch Output tensor: {torch_output_tensor.shape}")

    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    logger.info(assert_with_pcc(torch_output_tensor, output_tensor, 0.98))


@pytest.mark.parametrize(
    "N, C, H, W, num_groups",
    [
        # (1, 64, 192, 624, 16),
        # (1, 64, 96, 312, 16),
        # (1, 128, 48, 156, 16),
        # (1, 256, 24, 78, 16),
        # (1, 512, 12, 39, 16),
        # (1, 256, 48, 156, 16),
        # (1, 256, 12, 39, 16),
        # (1, 256, 159, 159, 16),
    ],
)
def test_group_norm_with_height_sharded_oft_padded(device, N, C, H, W, num_groups):
    torch.manual_seed(0)

    grid_size = ttnn.CoreGrid(y=1, x=2)

    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias
    )
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_tensor = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    print(f"before: {input_tensor.shape}")

    input_tensor = ttnn.pad(input_tensor, [(0, 0), (0, 0), (0, 32 - (input_tensor.shape[2] % 32)), (0, 0)], 0)

    print(f"after: {input_tensor.shape}")

    # input mask
    input_mask_tensor = ttnn.create_group_norm_input_mask(C, num_groups, grid_size.y)
    input_mask_tensor = ttnn.from_torch(
        input_mask_tensor,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    gamma = ttnn.create_group_norm_weight_bias_rm(torch_weight, C, grid_size.y)
    beta = ttnn.create_group_norm_weight_bias_rm(torch_bias, C, grid_size.y)

    gamma_t = ttnn.from_torch(
        gamma,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_t = ttnn.from_torch(
        beta,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.group_norm(
        input_tensor,
        num_groups=num_groups,
        input_mask=input_mask_tensor,
        weight=gamma_t,
        bias=beta_t,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        core_grid=grid_size,
        inplace=False,
    )

    output_tensor = ttnn.to_memory_config(output_tensor, ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


import math

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "N, C, H, W, num_groups, shard, cores_y, cores_x",
    [
        (1, 256, 12, 40, 16, "BS", 4, 5),
        (1, 256, 24, 80, 16, "HS", 4, 5),
        (1, 256, 48, 160, 16, "HS", 4, 5),
        (1, 512, 12, 40, 16, "BS", 4, 5),
        (1, 64, 96, 320, 16, "HS", 4, 5),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_group_norm_oft_tt(device, N, C, H, W, num_groups, shard, cores_y, cores_x):
    assert C % num_groups == 0, "Number of channels must be divisible by number of groups"

    # this check maybe is not in your version yet so instead we will use direct setup
    # compute_grid = device.compute_with_storage_grid_size()
    # if compute_grid.x != 5 or compute_grid.y != 4:
    # pytest.skip(f"Test requires compute grid size of 5x4, but got {compute_grid.x}x{compute_grid.y}")
    grid_size = ttnn.CoreGrid(y=cores_y, x=cores_x)
    # Generate torch tensor
    torch.manual_seed(0)
    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)
    # Execute torch group_norm
    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias, eps=1e-5
    )
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_tensor = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Generate input mask
    if shard == "HS":
        grid_x = grid_size.x * grid_size.y
        grid_y = 1
    else:
        grid_x = grid_size.x
        grid_y = grid_size.y
    input_mask_tensor = ttnn.create_group_norm_input_mask(C, num_groups, grid_y)
    input_mask_tensor = ttnn.from_torch(
        input_mask_tensor,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Generate gamma/beta tensors
    gamma = ttnn.create_group_norm_weight_bias_rm(torch_weight, C, grid_y)
    beta = ttnn.create_group_norm_weight_bias_rm(torch_bias, C, grid_y)

    gamma_t = ttnn.from_torch(
        gamma,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_t = ttnn.from_torch(
        beta,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Generate shard config
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = (H * W) // grid_x, C // grid_y
    if shard == "HS":
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        sharded_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
    elif shard == "BS":
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
        sharded_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
    input_tensor = ttnn.to_memory_config(input_tensor, memory_config=sharded_mem_config)

    output_tensor = ttnn.group_norm(
        input_tensor,
        num_groups=num_groups,
        input_mask=input_mask_tensor,
        weight=gamma_t,
        bias=beta_t,
        memory_config=sharded_mem_config,
        core_grid=grid_size,
        epsilon=1e-5,
    )
    output_tensor = ttnn.to_memory_config(output_tensor, ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


def _nearest_32_per_core(x, core):
    return math.ceil(x / core / 32) * 32 * core


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize(
    "N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x",
    [
        ### oft 384*1248
        (1, 64, 192, 624, 16, 10, 2, 4),  # 0.999
        (1, 64, 96, 312, 16, 10, 2, 5),  # 0.999
        (1, 256, 159, 159, 16, 4, 4, 4),  # 0.999
        (1, 128, 48, 156, 16, 4, 4, 4),  # 0.999
        (1, 256, 48, 156, 16, 4, 4, 4),  # 0.999
        (1, 256, 24, 78, 16, 4, 4, 4),  # 0.999
        (1, 256, 12, 39, 16, 4, 1, 5),  # 0.999
        (1, 512, 12, 39, 16, 4, 1, 5),  # 0.999
    ],
)
def test_group_norm_DRAM_oft_tt(device, N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x):
    compute_grid = device.compute_with_storage_grid_size()
    if compute_grid.x != 5 or compute_grid.y != 4:
        pytest.skip(f"Test requires compute grid size of 5x4, but got {compute_grid.x}x{compute_grid.y}")

    torch.manual_seed(0)
    grid_size = ttnn.CoreGrid(y=cores_y, x=cores_x)
    # torch input tensor
    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias
    )
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    # input tensor
    input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_tensor_row_major = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    print(
        f"input_tensor_row_major shape: {input_tensor_row_major.shape} padded shape: {input_tensor_row_major.padded_shape}"
    )
    unpadded_shape = input_tensor_row_major.shape
    out_shape = [
        unpadded_shape[0],
        unpadded_shape[1],
        _nearest_32_per_core(unpadded_shape[2], cores_x),
        _nearest_32_per_core(unpadded_shape[3], cores_y),
    ]
    print(f"unpadded_shape: {unpadded_shape} out_shape: {out_shape}")
    input_tensor_tilized = ttnn.tilize_with_val_padding(
        input_tensor_row_major, output_tensor_shape=out_shape, pad_value=0, use_multicore=True
    )
    print(input_tensor_tilized)
    print(f"input_tensor_tilized shape: {input_tensor_tilized.shape} padded shape: {input_tensor_tilized.padded_shape}")
    input_mask_tensor = ttnn.create_group_norm_input_mask(C, num_groups, grid_size.y)
    input_mask_tensor = ttnn.from_torch(
        input_mask_tensor,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # gamma/beta
    gamma = ttnn.create_group_norm_weight_bias_rm(torch_weight, C, grid_size.y)
    beta = ttnn.create_group_norm_weight_bias_rm(torch_bias, C, grid_size.y)
    gamma_t = ttnn.from_torch(
        gamma,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_t = ttnn.from_torch(
        beta,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # groupnorm
    output_tensor = ttnn.group_norm(
        input_tensor_tilized,
        num_groups=num_groups,
        input_mask=input_mask_tensor,
        weight=gamma_t,
        bias=beta_t,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_layout=ttnn.TILE_LAYOUT,
        core_grid=grid_size,
        inplace=False,
        num_out_blocks=num_out_blocks,
        epsilon=1e-5,
    )

    ttnn.synchronize_device(device)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    logger.info(assert_with_pcc(torch_output_tensor, output_tensor[:, :, : H * W, :C], 0.9994))
