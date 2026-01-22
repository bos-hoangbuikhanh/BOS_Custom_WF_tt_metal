import math

from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d, preprocess_model_parameters

import ttnn
from models.bos_model.pdl.reference.panoptic_seg import Conv2d

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


def custom_preprocessor(model, name):
    parameters = {}

    if isinstance(model, Conv2d):
        parameters["conv"] = {}

        if model.norm:
            weight, bias = fold_batch_norm2d_into_conv2d(model, model.norm)
        else:
            weight = model.weight
            bias = model.bias

        parameters["conv"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        if bias is not None:
            parameters["conv"]["bias"] = ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=ttnn.float32)

    return parameters


def create_pdl_model_parameters(model, input_tensor=None, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=None,
    )
    # parameters.conv_args = {}
    # parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)

    parameters["model_args"] = model

    return parameters


TILE_WIDTH = 32


def get_shard_grid_from_num_cores(device, ncores):
    device_grid = device.compute_with_storage_grid_size()
    max_grid_size = (device_grid.y, device_grid.x)
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
        raise ValueError("Invalid ncores")


def upsample_multicore_common(
    device,
    tt_input,
    scale_h,
    scale_w,
    shard_strategy=ttnn.ShardStrategy.HEIGHT,
    shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
    mode="bilinear",
    core_range=None,
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
):
    if use_signpost:
        signpost(header="sharded_upsample_multicore")

    ## input shape is N, H, W, C
    batch_size, height, width, num_channels = tt_input.shape

    TILE_WIDTH = 32

    num_channels_unpadded = num_channels
    num_channels_is_padded = False
    if num_channels % TILE_WIDTH != 0:
        num_channels_is_padded = True
        num_channels_pad_shape = TILE_WIDTH - num_channels % TILE_WIDTH
        num_channels = num_channels + num_channels_pad_shape

        # Pad the ttnn input tensor
        tt_input = ttnn.pad(tt_input, ((0, 0), (0, 0), (0, 0), (0, num_channels_pad_shape)), value=0)

    scale_factor = (scale_h, scale_w)

    num_bytes = 2  ## only BFLOAT16 is supported

    ## calculate ncores, corresponding grid_size and in_shard_shape based on the input_shape
    ncores = None
    shard_grid = None
    device_grid = device.compute_with_storage_grid_size()
    max_grid_size = (device_grid.y, device_grid.x)
    if core_range != None:
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(core[0][0], core[0][1]), ttnn.CoreCoord(core[1][0], core[1][1]))
                for core in core_range
            }
        )
        if shard_strategy == ttnn.ShardStrategy.BLOCK:
            if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
                ncores = (core_range[0][1][0] - core_range[0][0][0] + 1, core_range[0][1][1] - core_range[0][0][1] + 1)
            elif shard_orientation == ttnn.ShardOrientation.COL_MAJOR:
                ncores = (core_range[0][1][1] - core_range[0][0][1] + 1, core_range[0][1][0] - core_range[0][0][0] + 1)
        elif shard_strategy == ttnn.ShardStrategy.HEIGHT:
            ncores = shard_grid.num_cores()
        else:
            raise ValueError("Invalid shard strategy")

    else:
        if shard_strategy == ttnn.ShardStrategy.HEIGHT:
            max_nshards = min(batch_size * height * width, max_grid_size[0] * max_grid_size[1])
            nshards = max_nshards
            if mode == "bilinear":
                # For bilinear, sticks per core must be divisible by width
                while nshards > 0:
                    if batch_size * height % nshards == 0 and (batch_size * height * width // nshards) % width == 0:
                        break
                    nshards -= 1
            else:
                # For nearest, just need total elements divisible by nshards
                while nshards > 0:
                    if batch_size * height * width % nshards == 0:
                        break
                    nshards -= 1
            ncores = nshards
        elif shard_strategy == ttnn.ShardStrategy.BLOCK:
            max_nshards_h = min(batch_size * height * width, max_grid_size[0])  ## height along NHW
            max_nshards_w = min(num_channels, max_grid_size[1])  ## width along C
            ## find nshards_h along NHW
            nshards_h = max_nshards_h
            while nshards_h > 0:
                if batch_size * height % nshards_h == 0:
                    break
                nshards_h -= 1
            ## find nshards_w along C
            nshards_w = max_nshards_w
            while nshards_w > 0:
                ## make sure: 1. nshards_w divides num_channels, and 2. shard_shape[1] is aligned to 32B
                if num_channels % nshards_w == 0 and math.ceil(num_channels * num_bytes / nshards_w) % TILE_WIDTH == 0:
                    break
                nshards_w -= 1
            if nshards_w == 0 or nshards_h == 0:
                raise ValueError("nshards_h or nshards_w is 0")
            ncores = (nshards_h, nshards_w)
        shard_grid = get_shard_grid_from_num_cores(device, ncores)

    if shard_strategy == ttnn.ShardStrategy.BLOCK:
        tensor_memory_layout = ttnn.types.TensorMemoryLayout.BLOCK_SHARDED
    elif shard_strategy == ttnn.ShardStrategy.HEIGHT:
        tensor_memory_layout = ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED

    ## input shard
    if shard_strategy == ttnn.ShardStrategy.BLOCK:
        shard_height = math.ceil(batch_size * height * width / ncores[0])
        shard_width = math.ceil(num_channels / ncores[1])
    elif shard_strategy == ttnn.ShardStrategy.HEIGHT:
        shard_height = math.ceil(batch_size * height * width / ncores)
        shard_width = num_channels
    shard_shape = (shard_height, shard_width)
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    in_sharded_mem_config = ttnn.MemoryConfig(tensor_memory_layout, ttnn.types.BufferType.L1, shard_spec)

    ## output shard
    shard_height = shard_height * scale_h * scale_w
    shard_shape = (shard_height, shard_width)
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    out_sharded_mem_config = ttnn.MemoryConfig(tensor_memory_layout, ttnn.types.BufferType.L1, shard_spec)

    scale_factor = (scale_h, scale_w)

    input_tensor = ttnn.to_memory_config(tt_input, memory_config=in_sharded_mem_config)
    if mode == "bilinear":
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=math_fidelity,
            math_approx_mode=math_approx_mode,
            fp32_dest_acc_en=False,
        )
        output_tensor = ttnn.upsample(
            input_tensor,
            scale_factor,
            mode="bilinear",
            memory_config=out_sharded_mem_config,
            compute_kernel_config=compute_kernel_config,
        )
    else:
        output_tensor = ttnn.upsample(input_tensor, scale_factor, memory_config=out_sharded_mem_config)

    output_tensor = ttnn.to_memory_config(output_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if num_channels_is_padded:
        output_tensor = output_tensor[:, :, :, :num_channels_unpadded]
    return output_tensor


def sharded_concat(
    input_tensors, num_cores=20, dim=3, skip_s2i=False, use_output_dram=False
):  # expected input tensors to be in fp16, RM, same (h*w)
    if use_signpost:
        signpost(header="sharded_concat")

    shard_height = (input_tensors[0].shape[2] + num_cores - 1) // num_cores

    input_sharded_memory_configs = []

    for i in range(len(input_tensors)):
        input_sharded_memory_config = ttnn.create_sharded_memory_config(
            (shard_height, input_tensors[i].shape[-1]),
            core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 3))}),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        input_sharded_memory_configs.append(input_sharded_memory_config)

    sharded_inputs = [
        ttnn.to_memory_config(tensor, config) for tensor, config in zip(input_tensors, input_sharded_memory_configs)
    ]

    total_width = sum(tensor.shape[-1] for tensor in input_tensors)
    out_sharded_memory_config = ttnn.create_sharded_memory_config(
        (shard_height, total_width),
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 3))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )

    output = ttnn.concat(sharded_inputs, dim, memory_config=out_sharded_memory_config)
    if not skip_s2i:
        if use_output_dram:
            output = ttnn.sharded_to_interleaved(output, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            output = ttnn.sharded_to_interleaved(output, memory_config=ttnn.L1_MEMORY_CONFIG)

    return output


def tt_to_torch_tensor(tt_tensor):
    tt_output = tt_tensor.cpu().to(ttnn.ROW_MAJOR_LAYOUT)
    return tt_output.to_torch()
