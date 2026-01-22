import torch
import pytest
import ttnn
from models.common.utility_functions import divup
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal
import math

# torch.set_printoptions(threshold=float('inf'), linewidth=200)


### Math operations ###
def _nearest_32(x):
    return math.ceil(x / 32) * 32


def setup_l1_sharded_input(device, torch_input_tensor=None, min_channels=16, num_cores=18):
    if num_cores == 20:
        core_grid = ttnn.CoreGrid(y=4, x=5)
    elif num_cores == 16:
        core_grid = ttnn.CoreGrid(y=4, x=4)
    elif num_cores == 18:
        core_grid = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(4, 2),
                ),
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 3),
                    ttnn.CoreCoord(2, 3),
                ),
            ]
        )
    else:
        core_grid = ttnn.CoreGrid(y=8, x=8)

    n, h, w, c = torch_input_tensor.shape
    if c < min_channels:
        channel_padding_needed = min_channels - c
        torch_input_tensor = torch.nn.functional.pad(
            torch_input_tensor, (0, channel_padding_needed, 0, 0, 0, 0), value=0.0
        )
        c = min_channels
    # torch_input_tensor = torch_input_tensor.reshape(1, 1, n * h * w, c)
    nhw = n * h * w
    shard_size = _nearest_32(nhw / num_cores)
    input_mem_config = ttnn.create_sharded_memory_config(
        [1, 1, shard_size, c],
        core_grid,
        ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    return tt_inputs_host, input_mem_config


def setup_bos_dram_sharded_input(device, torch_input_tensor=None, mesh_mapper=None, mesh_composer=None):
    tt_inputs_host, input_mem_config = setup_l1_sharded_input(device, torch_input_tensor)
    dram_grid_size = device.dram_grid_size()
    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
        ),
        [
            divup(tt_inputs_host.volume() // tt_inputs_host.shape[-1], dram_grid_size.x),
            tt_inputs_host.shape[-1],
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded_mem_config_DRAM = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
    )

    return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config


# 매핑: ttnn dtype → torch dtype
TTNN_TO_TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    # 추가적으로 다른 dtype들도 여기 넣을 수 있음
}


# @pytest.mark.parametrize("shape", [(1, 3, 256, 256), (1, 3, 512, 512), (1, 3, 320, 320), (1, 3, 384, 640) ]) #
@pytest.mark.parametrize("shape", [[1, 1, 23040, 64]])  #
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_sharding(device, shape, dtype):
    print("This is pytest for bos")

    torch.manual_seed(1231)

    torch_dtype = TTNN_TO_TORCH_DTYPE[dtype]  # convert
    source_tensor = torch.randn(*shape, dtype=torch_dtype)  # shape unpack
    tt_inputs_host, sharded_mem_config_DRAM = setup_l1_sharded_input(device=device, torch_input_tensor=source_tensor)
    sharding_result = tt_inputs_host.to(device, sharded_mem_config_DRAM)
    sharding_result = ttnn.to_layout(sharding_result, ttnn.TILE_LAYOUT)
    ret = ttnn.sharded_to_interleaved(sharding_result, ttnn.DRAM_MEMORY_CONFIG)
    input_tensor = ttnn.from_device(ret).to_torch()

    result, msg = assert_with_pcc(source_tensor, input_tensor)
