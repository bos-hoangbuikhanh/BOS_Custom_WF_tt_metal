import time
import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import (
    divup,
)


@pytest.mark.parametrize(
    "H, W, C",
    [
        (1024, 512, 1),
        (1280, 720, 1),
        (2048, 1024, 1),
        (2176, 960, 1),
    ],
    ids=["1024x512x1", "1280x720x1", "2048x1024x1", "2176x960x1"],  #
)
def test_tensor_read(device, H, W, C):
    core_grid = device.compute_with_storage_grid_size()
    num_cores = core_grid.x * core_grid.y
    # assert num_cores == 64

    torch_input_tensor = torch.randn(1, 1, C, H * W)
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16)

    tensor_on_device = ttnn.to_device(ttnn_input_tensor, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    outputs = []

    logger.info("Begin measurement")
    num_iters = 100
    start = time.time()
    for i in range(num_iters):
        outputs.append(ttnn.from_device(tensor_on_device, blocking=False))
    ttnn.synchronize_device(device)
    end = time.time()

    total_time = end - start
    total_size_in_bytes = 2 * C * H * W * num_iters  # 2 in for bfloat16
    total_size_in_GiB = total_size_in_bytes / (1024**3)
    time_per_tensor = total_time / num_iters
    logger.info(
        f"Average time per tensor read for resolution: {H}x{W}x{C}: {time_per_tensor*1000:.1f}ms samples_per_sec: {1/time_per_tensor:.1f} GiB/s = {total_size_in_GiB / total_time:.1f}"
    )


@pytest.mark.parametrize(
    "H, W, C",
    [
        (1024, 512, 3),
        (1280, 720, 3),
        (2048, 1024, 3),
        (2176, 960, 3),
    ],
    ids=["1024x512x1", "1280x720x1", "2048x1024x1", "2176x960x1"],
)
def test_tensor_write(device, H, W, C):
    core_grid = device.compute_with_storage_grid_size()
    num_cores = core_grid.x * core_grid.y
    # assert num_cores == 64

    torch_input_tensor = torch.randn(1, C, H, W)
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16)

    dram_grid_size = device.dram_grid_size()
    dram_num_cores = dram_grid_size.x * dram_grid_size.y
    dram_shard_grid = ttnn.num_cores_to_corerangeset(dram_num_cores, dram_grid_size)

    dram_shard_shape = [
        divup(C * H, dram_grid_size.x),
        W,
    ]
    dram_shard_spec = ttnn.ShardSpec(
        dram_shard_grid,
        dram_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    dram_height_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.DRAM,
        dram_shard_spec,
    )
    logger.info(f"DRAM Height Sharded Memory Config: {dram_height_sharded_mem_config}")
    tensor_on_device = ttnn.to_device(ttnn_input_tensor, device=device, memory_config=dram_height_sharded_mem_config)

    logger.info("Begin measurement")
    num_iters = 10000
    start = time.time()
    for i in range(num_iters):
        ttnn.copy_host_to_device_tensor(ttnn_input_tensor, tensor_on_device)
    ttnn.synchronize_device(device)
    end = time.time()

    total_time = end - start
    total_size_in_bytes = 2 * C * H * W * num_iters  # 2 in for bfloat16
    total_size_in_GiB = total_size_in_bytes / (1024**3)
    time_per_tensor = total_time / num_iters
    logger.info(
        f"Average time per tensor write for resolution: {H}x{W}x{C}: {time_per_tensor*1000:.1f}ms samples_per_sec: {1/time_per_tensor:.1f} GiB/s = {total_size_in_GiB / total_time:.1f}"
    )
