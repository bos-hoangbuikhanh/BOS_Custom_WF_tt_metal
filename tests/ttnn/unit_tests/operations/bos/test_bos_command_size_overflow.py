import torch
import pytest
import ttnn


def test_bos_sharded_from_torch_overflow(device):
    """
    Test to verify command size overflow fix for sharded buffers.

    This test replicates the issue found in Panoptic DeepLab where large sharded tensors
    would cause command size to exceed the max prefetch command size (131,072 B).

    Background:
    - Creates a [1, 3, 512, 1024] input tensor (similar to PDL preprocessing)
    - Pads channels to 8, converts to NHWC layout
    - Computes shard_height over a 5x4 (20) core grid
    - Builds HEIGHT sharded memory config
    - Calls ttnn.from_torch(..., memory_config=sharded)

    Expected behavior:
    - v63: No overflow (command size = 131,072 B exactly)
    - v64 without fix: Overflow (command size = 131,136 B, exceeds 131,072 B limit)
    - v64 with fix: No overflow (command size = 131,008 B)

    Root cause (v63 → v64):
    - CQDispatchCmd expanded from 16 B to CQDispatchCmdLarge (32 B) for 64-bit addressing
    - DeviceCommandCalculator adds additional alignment padding
    - calculate_num_pages_available_in_cq didn't account for this overhead

    Fix:
    - Add safety margin (pcie_alignment × 2) to prevent overflow
    - This reduces num_pages from 8,186 to 8,182, keeping command size under limit
    """
    torch.manual_seed(0)

    # PDL default input size
    batch_size, channels, height, width = 1, 3, 512, 1024
    torch_input = torch.randn(batch_size, channels, height, width, dtype=torch.bfloat16)

    # Match PDL preprocessing: pad channels to SHARD_WIDTH=8, convert to NHWC
    SHARD_WIDTH = 8
    pad_c = SHARD_WIDTH - channels
    if pad_c > 0:
        torch_input = torch.nn.functional.pad(torch_input, (0, 0, 0, 0, 0, pad_c), mode="constant", value=0)
    torch_input = torch_input.permute(0, 2, 3, 1)

    # Compute shard shape across full compute grid (typically 5x4 = 20 cores)
    core_grid = device.compute_with_storage_grid_size()
    num_cores = core_grid.x * core_grid.y
    HW = height * width
    shard_height = (1 * HW + num_cores - 1) // num_cores

    core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1))}
    )

    sharded_memory_config = ttnn.create_sharded_memory_config_(
        shape=(shard_height, SHARD_WIDTH),
        core_grid=core_range_set,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # This call triggers a large sharded buffer write that could cause command size overflow
    # In v64 without fix: would cause "Generated prefetcher command of size 131136 B exceeds max 131072 B"
    # With fix: command size stays within limit (131,008 B or less)
    tt_tensor = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=sharded_memory_config,
    )

    # Verify the tensor was created successfully (no overflow/hang occurred)
    assert tt_tensor is not None
    print(f"[BOS_TEST] Successfully created sharded tensor with shape: {tt_tensor.shape}")
