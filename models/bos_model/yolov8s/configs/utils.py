import math

import ttnn
from models.bos_model.yolov8s.utilities.utility_functions import _nearest_32


def ConvConfig(
    core_grid=None,
    core_start=(0, 0),
    core_end=(3, 3),
    shard_layout="Height",
    dtype=ttnn.bfloat16,
    weights_dtype=ttnn.bfloat16,
    math_fidelity=ttnn.MathFidelity.LoFi,
    use_shallow_conv_variant=False,
    fp32_dest_acc_enabled=False,
    activation="",
    deallocate_activation=False,
    reshard_if_not_optimal=False,
    transpose_shards=False,
    act_block_h_override=0,
    reallocate_halo_output=False,
    output_layout=ttnn.TILE_LAYOUT,
):
    map_shard = {
        "Height": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "Width": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        "Block": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "height": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "width": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        "block": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    }

    if core_grid is None:
        core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(core_start, core_end)})
    # print(map_shard[shard_layout])
    # print(output_layout)
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=weights_dtype,
        # activation=None
        deallocate_activation=deallocate_activation,
        reallocate_halo_output=reallocate_halo_output,
        # config_tensors_in_dram=False,
        act_block_h_override=act_block_h_override,
        # act_block_w_div=1,
        reshard_if_not_optimal=reshard_if_not_optimal,
        # override_sharding_config=False,
        shard_layout=map_shard[shard_layout],
        core_grid=core_grid,
        transpose_shards=transpose_shards,
        # output_layout=output_layout,
        enable_act_double_buffer=False,
        # enable_weights_double_buffer=False,
        # full_inner_dim=False,
        # in_place=False,
        # enable_kernel_stride_folding=False,
        # enable_activation_reuse=False,
        # force_split_reader=None,
    )

    return conv_config


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


def UpsampleConfig(
    input_shape=[0, 0, 0, 0],
    scale_h=2,
    scale_w=2,
    shard_strategy="Height",
    shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
    max_grid_size=(4, 4),
):
    batch_size, num_channels, height, width = input_shape
    num_bytes = 2  ## only BFLOAT16 is supported

    map_shard_layout = {
        "Height": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "Block": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "height": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "block": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    }

    ncores = None
    if map_shard_layout[shard_strategy] == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        max_nshards = min(batch_size * height, max_grid_size[0] * max_grid_size[1])
        nshards = max_nshards
        while nshards > 0:
            if batch_size * height % nshards == 0:
                break
            nshards -= 1
        ncores = nshards
    elif map_shard_layout[shard_strategy] == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        max_nshards_h = min(batch_size * height, max_grid_size[0])  ## height along NHW
        max_nshards_w = min(num_channels, max_grid_size[1])  ## width along C
        nshards_h = max_nshards_h
        while nshards_h > 0:
            if batch_size * height % nshards_h == 0:
                break
            nshards_h -= 1
        nshards_w = max_nshards_w
        while nshards_w > 0:
            if num_channels % nshards_w == 0 and math.ceil(num_channels * num_bytes / nshards_w) % 32 == 0:
                break
            nshards_w -= 1
        if nshards_w == 0 or nshards_h == 0:
            raise ValueError("nshards_h or nshards_w is 0")
        ncores = (nshards_h, nshards_w)

    shard_grid = get_shard_grid_from_num_cores(max_grid_size, ncores)

    if map_shard_layout[shard_strategy] == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        shard_height = math.ceil(batch_size * height * width / ncores[0])
        shard_width = math.ceil(num_channels / ncores[1])
    elif map_shard_layout[shard_strategy] == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        shard_height = math.ceil(batch_size * height * width / ncores)
        shard_width = num_channels

    shard_height = shard_height * scale_h * scale_w
    shard_shape = (shard_height, shard_width)
    # print(shard_orientation)
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)

    out_sharded_mem_config = ttnn.MemoryConfig(map_shard_layout[shard_strategy], ttnn.types.BufferType.L1, shard_spec)

    return out_sharded_mem_config


def ConcatConfig(
    input_shape=[0, 0, 0, 0],
    shard_strategy="Height",
    shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
    max_grid_size=(4, 4),
):
    _, _, nhw, channels = input_shape
    num_bytes = 2  ## only BFLOAT16 is supported

    map_shard_layout = {
        "Height": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "Block": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "height": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "block": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    }

    ncores = max_grid_size[0] * max_grid_size[1]
    if map_shard_layout[shard_strategy] == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        shard_height = _nearest_32(nhw // (max_grid_size[0] * max_grid_size[1]))
        shard_width = channels
    else:
        raise Exception("Only supports Height Sharding for now")

    shard_grid = get_shard_grid_from_num_cores(max_grid_size, ncores)

    shard_shape = (shard_height, shard_width)
    # print(shard_orientation)
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)

    out_sharded_mem_config = ttnn.MemoryConfig(map_shard_layout[shard_strategy], ttnn.types.BufferType.L1, shard_spec)

    return out_sharded_mem_config
