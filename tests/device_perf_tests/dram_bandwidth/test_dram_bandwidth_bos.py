import ttnn
import torch
import pytest

import math


def create_ws_memory_config(output_core_grid, input_shape):
    if isinstance(output_core_grid, tuple):
        output_core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(output_core_grid[0] - 1, output_core_grid[1] - 1)),
            ]
        )
    else:
        output_core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(x, y),
                    ttnn.CoreCoord(x, y),
                )
                for x, y in output_core_grid
            ]
        )
    padded_out_w = math.ceil(input_shape[3] / output_core_range_set.num_cores() / 32) * 32
    output_memory_config = ttnn.create_sharded_memory_config(
        shape=(
            input_shape[0] * input_shape[1] * input_shape[2],
            padded_out_w,
        ),
        core_grid=output_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    return output_memory_config


@pytest.mark.parametrize(
    "shape, core_grid",
    [
        # ([1, 1, 32, 32 * 64], (8, 8)),
        # ([1, 1, 5 * 1024 * 1024 // 32 // 64, 32 * 64], (8, 8)),
        # ([1, 1, 10 * 1024 * 1024 // 32 // 64, 32 * 64], (8, 8)),
        # ([1, 1, 20 * 1024 * 1024 // 32 // 64, 32 * 64], (8, 8)),
        # ([1, 1, 5 * 1024 * 1024 // 32 // 32, 32 * 32], (4, 8)),
        # ([1, 1, 10 * 1024 * 1024 // 32 // 32, 32 * 32], (4, 8)),
        # ([1, 1, 5 * 1024 * 1024 // 32 // 32, 32 * 32], (8, 4)),
        # ([1, 1, 10 * 1024 * 1024 // 32 // 32, 32 * 32], (8, 4)),
        # ([1, 1, 5 * 512 * 1024 // 20 // 32, 20 * 32], (4, 4)),
        ([1, 1, 5 * 512 * 1024 // 20 // 32, 20 * 32], (5, 4)),
        #    ([1, 1, 8 * 512 * 1024 // 20 // 32, 20 * 32], (5, 4)),
        ([1, 1, 10 * 512 * 1024 // 20 // 32, 20 * 32], (5, 4)),
        ([1, 1, 15 * 512 * 1024 // 20 // 32, 20 * 32], (5, 4)),
        # ([1, 1, 10 * 512 * 1024 // 20 // 32, 20 * 32], (4, 4)),
        # ([1, 1, 10 * 512 * 1024 // 20 // 32, 20 * 32], (5, 4)),
    ],
)
def test_dram_interleaved(
    device,
    shape,
    core_grid,
):
    grid = device.compute_with_storage_grid_size()
    # assert grid.x * grid.y == 64, "Only valid on 64 core grid"

    torch_input_tensor = torch.randint(low=0, high=10, size=shape, dtype=torch.int32)
    ttnn_input_tensor_dram_interleaved = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_ws_memory_config = create_ws_memory_config(core_grid, shape)

    ttnn_input_tensor_ws = ttnn.to_memory_config(
        ttnn_input_tensor_dram_interleaved, memory_config=output_ws_memory_config
    )

    print(ttnn_input_tensor_dram_interleaved.memory_config())
    print(ttnn_input_tensor_ws.memory_config())
