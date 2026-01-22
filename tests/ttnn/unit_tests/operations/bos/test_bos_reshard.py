import torch
import pytest
import ttnn
from models.common.utility_functions import divup
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal
from models.common.utility_functions import skip_for_grayskull, skip_for_wormhole_b0
import math

ALIGNMENT = 16  # Define alignment constant

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def setup_l1_sharded_input(device, torch_input_tensor=None, min_channels=32, num_cores=16):
    """
    Sets up a L1-sharded memory config for testing.

    Args:
        device: The target device for the tensor.
        torch_input_tensor (torch.Tensor): Input tensor in PyTorch format.
        min_channels (int): Minimum number of channels for alignment.
        num_cores (int): Number of cores in the core grid.

    Returns:
        tuple: A tuple containing the sharded tensor and its memory configuration.
    """

    def align(x):
        return math.ceil(x / min_channels) * min_channels

    # Define core grid based on the number of cores
    if num_cores == 20:
        core_grid = ttnn.CoreGrid(y=4, x=5)
    elif num_cores == 16:
        core_grid = ttnn.CoreGrid(y=4, x=4)
    else:
        core_grid = ttnn.CoreGrid(y=8, x=8)

    # Convert tensor layout from NCHW to NHWC
    torch_input_tensor = torch_input_tensor.permute((0, 2, 3, 1))
    n, h, w, c = torch_input_tensor.shape
    torch_input_tensor = torch_input_tensor.reshape(1, 1, n * h * w, c)

    # Align shard size
    shard_size = align(torch_input_tensor.shape[2] / num_cores)

    # Create sharded memory configuration
    input_mem_config = ttnn.create_sharded_memory_config(
        shape=(
            1,
            1,
            shard_size,
            min_channels if torch_input_tensor.shape[3] < min_channels else align(torch_input_tensor.shape[3]),
        ),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Pad the input tensor to ensure alignment
    tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.Layout.ROW_MAJOR)
    tt_inputs_host = ttnn.pad(
        tt_inputs_host,
        [
            tt_inputs_host.shape[0],
            tt_inputs_host.shape[1],
            tt_inputs_host.shape[2],
            min_channels if c < min_channels else align(c),
        ],
        [0, 0, 0, 0],
        0,
    )

    return tt_inputs_host, input_mem_config


def setup_bos_dram_sharded_input(device, torch_input_tensor=None, mesh_mapper=None, mesh_composer=None):
    """
    Sets up a DRAM-sharded memory config for testing.

    Args:
        device: The target device for the tensor.
        torch_input_tensor (torch.Tensor): Input tensor in PyTorch format.
        mesh_mapper: Optional mesh mapper for tensor placement.
        mesh_composer: Optional mesh composer for tensor placement.

    Returns:
        tuple: A tuple containing the sharded tensor, DRAM memory configuration, and input memory configuration.
    """

    def divup(a, b):
        return (a + b - 1) // b

    # Setup L1 sharded input
    tt_inputs_host, input_mem_config = setup_l1_sharded_input(
        device, torch_input_tensor, min_channels=ALIGNMENT, num_cores=20
    )

    # Define DRAM grid size and shard specification
    dram_grid_size = device.dram_grid_size()
    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
        ),
        [
            divup(tt_inputs_host.volume() // tt_inputs_host.padded_shape[-1], (dram_grid_size.x * dram_grid_size.y)),
            tt_inputs_host.padded_shape[-1],
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    # Create DRAM memory configuration
    sharded_mem_config_DRAM = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
    )

    return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config


# Mapping: ttnn dtype → torch dtype
TTNN_TO_TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    # Additional dtypes can be added here
}


@pytest.mark.parametrize(
    "shape",
    [
        (2, 3, 32, 32),
        (1, 3, 256, 256),
        (1, 3, 512, 512),
        (1, 3, 320, 320),
        (3, 3, 384, 640),
        (4, 128, 128, 32),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_dram_shard_loopback(device, shape, dtype):
    """
    Tests the loopback functionality for DRAM-sharded tensors.

    Args:
        device: The target device for the tensor.
        shape (tuple): Shape of the input tensor.
        dtype: Data type of the tensor.
    """
    print("This is pytest for DRAM sharding")
    ttnn.device.EnableMemoryReports()
    torch.manual_seed(1231)

    # Generate source tensor
    b, c, h, w = shape
    torch_dtype = TTNN_TO_TORCH_DTYPE[dtype]
    source_pt_tensor = torch.randn(*shape, dtype=torch_dtype)

    # Setup DRAM-sharded input
    host_tt_tensor, sharded_mem_config_DRAM, _ = setup_bos_dram_sharded_input(
        device=device, torch_input_tensor=source_pt_tensor
    )

    # Transfer host tensor to DRAM sharding memory configuration, then transfer back to host
    # Ensure DRAM sharding is correct
    dram_sharded_tt_tensor = host_tt_tensor.to(device, sharded_mem_config_DRAM)
    output_pt_tensor = ttnn.from_device(dram_sharded_tt_tensor).to_torch()

    # Reshape source tensor for comparison
    source_pt_tensor = source_pt_tensor.permute((0, 2, 3, 1))  # NCHW -> NHWC
    source_pt_tensor = source_pt_tensor.reshape(1, 1, b * h * w, c)

    # Assert tensors are equal using PCC
    assert_with_pcc(source_pt_tensor, output_pt_tensor)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 32, 64, 64),
        (3, 3, 32, 32),
        (1, 3, 256, 256),
        (3, 3, 256, 256),
        (4, 3, 256, 256),
        (1, 3, 320, 320),
        (1, 115, 115, 16),
        (4, 115, 115, 16),
        (1, 3, 384, 640),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_dram_reshard_loopback(device, shape, dtype):
    """
    Tests the resharing functionality for ttnn.bos_reshard() DRAM shard -> L1 shard.

    Args:
        device: The target device for the tensor.
        shape (tuple): Shape of the input tensor.
        dtype: Data type of the tensor.
    """
    print("This is pytest for ttnn.bos_reshard() DRAM shard -> L1 shard")
    ttnn.device.EnableMemoryReports()
    torch.manual_seed(327)

    # Generate source tensor
    b, c, h, w = shape
    torch_dtype = TTNN_TO_TORCH_DTYPE[dtype]
    source_tensor = torch.randn(*shape, dtype=torch_dtype)

    # Setup DRAM-sharded input
    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = setup_bos_dram_sharded_input(
        device=device, torch_input_tensor=source_tensor
    )

    # Reshard tensor and transfer back to host
    tt_image_res = tt_inputs_host.to(device, sharded_mem_config_DRAM)
    output_tensor = ttnn.bos_reshard(tt_image_res, input_mem_config)
    output_tensor = ttnn.from_device(output_tensor).to_torch()

    # Reshape source tensor for comparison
    source_tensor = source_tensor.permute((0, 2, 3, 1))  # NCHW -> NHWC
    source_tensor = source_tensor.reshape(1, 1, b * h * w, c)

    # Assert tensors are equal using PCC
    assert_with_pcc(source_tensor, output_tensor)
