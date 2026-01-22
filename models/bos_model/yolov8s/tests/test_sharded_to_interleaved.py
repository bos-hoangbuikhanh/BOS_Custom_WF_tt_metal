import pytest
import torch

import ttnn
from models.bos_model.yolov8s.utilities.utility_functions import (
    _nearest_32,
    comp_pcc,
    load_resize_and_pad_channels,
    setup_l1_sharded_input,
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)


def get_shard_grid_from_num_cores(max_grid_size, ncores):
    if isinstance(ncores, int):
        if ncores % max_grid_size[1] == 0:
            core_grid = ttnn.CoreGrid(y=ncores // max_grid_size[1], x=max_grid_size[1])
            grid_coord = ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1)
            return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        else:
            if ncores < max_grid_size[1]:
                core_grid = ttnn.CoreGrid(y=1, x=ncores)
                grid_coord = ttnn.CoreCoord(core_grid.x - 1, 0)
                return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
            else:
                core_grid_1 = ttnn.CoreGrid(y=ncores // max_grid_size[1], x=max_grid_size[1])
                core_grid_2 = ttnn.CoreGrid(y=ncores // max_grid_size[1] + 1, x=ncores % max_grid_size[1])
                grid_coord_1 = ttnn.CoreCoord(core_grid_1.x - 1, core_grid_1.y - 1)
                grid_coord_2 = ttnn.CoreCoord(core_grid_2.x - 1, core_grid_2.y - 1)
                return ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord_1),
                        ttnn.CoreRange(ttnn.CoreCoord(0, grid_coord_2.y), grid_coord_2),
                    }
                )
    elif isinstance(ncores, tuple):
        ncores_h, ncores_w = ncores
        assert ncores_h <= max_grid_size[0]
        assert ncores_w <= max_grid_size[1]
        return ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(ncores_w - 1, ncores_h - 1),
                )
            }
        )
    else:
        raise ValueError("Invalid ncores", ncores)


@pytest.mark.parametrize(
    "layout",
    [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
    # [ttnn.ROW_MAJOR_LAYOUT],
    # [ttnn.TILE_LAYOUT],
)
@pytest.mark.parametrize(
    "input_shape, shard_scheme, shard_shape, core_range_set",
    [
        # ([1, 1, 1600, 64], ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (96, 64), [[(0,0),(4,2)],[(0,3),(1,3)]]),
        # ([1, 1, 1600, 80], ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (96, 96), [[(0,0),(4,2)],[(0,3),(1,3)]]),
        # ([1, 1, 400, 64], ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (32, 64), [[(0,0),(4,1)],[(0,2),(2,2)]]),
        # ([1, 1, 400, 80], ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (32, 96), [[(0,0),(4,1)],[(0,2),(2,2)]]),
        ([1, 1, 100, 64], ttnn.TensorMemoryLayout.BLOCK_SHARDED, (32, 32), [[(0, 0), (1, 3)]]),
        ([1, 1, 100, 80], ttnn.TensorMemoryLayout.BLOCK_SHARDED, (32, 32), [[(0, 0), (3, 3)]]),  # Fails
        (
            [1, 1, 100, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (32, 64),
            [[(0, 0), (4, 0)]],
        ),  # Working, i.e., reshard to solve - Not necessary tho
        (
            [1, 1, 100, 80],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (32, 96),
            [[(0, 0), (3, 0)]],
        ),  # Working, i.e., reshard to solve
        # ([1, 1, 1024, 64], ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (64, 64), [[(0,0),(4,2)],[(0,3),(0,3)]]),
        # ([1, 1, 1024, 80], ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (64, 96), [[(0,0),(4,2)],[(0,3),(0,3)]]),
        # ([1, 1, 256, 64], ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (32, 64), [[(0,0),(4,0)],[(0,1),(2,1)]]),
        # ([1, 1, 256, 80], ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (32, 96), [[(0,0),(4,0)],[(0,1),(2,1)]]),
        ([1, 1, 64, 64], ttnn.TensorMemoryLayout.BLOCK_SHARDED, (32, 32), [[(0, 0), (1, 1)]]),
        ([1, 1, 64, 80], ttnn.TensorMemoryLayout.BLOCK_SHARDED, (32, 32), [[(0, 0), (3, 1)]]),  # Fails
        (
            [1, 1, 64, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (32, 64),
            [[(0, 0), (1, 0)]],
        ),  # Working, i.e., reshard to solve - Not necessary tho
        (
            [1, 1, 64, 80],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (32, 96),
            [[(0, 0), (1, 0)]],
        ),  # Working, i.e., reshard to solve
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    # [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
    [ttnn.ShardOrientation.ROW_MAJOR],
)
def test_sharded_rm(
    device,
    input_shape,
    shard_shape,
    shard_scheme,
    shard_orientation,
    layout,
    core_range_set,
):
    # device_dict = {"device_id": 0, "l1_small_size": 20480, "trace_region_size": 10419200, "num_command_queues": 2}
    # ttnn_device = ttnn._ttnn.device
    # device = ttnn_device.CreateDevice(**device_dict)

    input_size = torch.Size(input_shape)
    x = torch.arange(input_size.numel()).reshape(input_size).bfloat16().float()

    xt = ttnn.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    ).to(
        device,
        ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.DRAM,
        ),
    )

    print(xt.shape)
    print(xt.memory_config())
    print(xt.get_layout())
    print()

    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(cores[0], cores[1]) for cores in core_range_set})

    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_memory_config = ttnn.MemoryConfig(shard_scheme, ttnn.types.BufferType.L1, shard_spec)

    yt = ttnn.to_memory_config(xt, memory_config=input_memory_config)
    yt = ttnn.to_layout(yt, layout)
    print(yt.shape)
    print(yt.memory_config())
    print(yt.get_layout())
    print()

    zt = ttnn.sharded_to_interleaved(
        yt,
        ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.L1,
        ),
        is_l1_aligned=True,
    )
    zt = ttnn.to_layout(zt, ttnn.TILE_LAYOUT)
    print(zt.shape)
    print(zt.memory_config())
    print(zt.get_layout())
    print()

    tt_og = xt.cpu().to_torch()

    tt_got_back = zt.cpu().to_torch()

    passing, output = comp_pcc(tt_og, tt_got_back)
    # logger.info(output)

    assert passing
