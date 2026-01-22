import ttnn
import torch

@ttnn.register_python_operation(
    name="ttnn.bos_create_bilinear_hash"
) 
def bos_create_bilinear_hash(device, step_x=100, step_y=100, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """
    Create a bilinear interpolation weight lookup table (hash) for 2D coordinates.

    The table precomputes bilinear interpolation weights for a uniform grid
    spanning [0, 1] in both x and y directions. Each grid cell stores the
    four bilinear weights corresponding to its corners:
        - w1: top-left
        - w2: top-right
        - w3: bottom-left
        - w4: bottom-right

    Args:
        device (ttnn.Device): Target device to allocate the hash table on.
        step_x (int, optional): Number of discretization steps along the x-axis.
            Defines resolution of the grid. Default is 100.
        step_y (int, optional): Number of discretization steps along the y-axis.
            Defines resolution of the grid. Default is 100.
        dtype (ttnn.dtype, optional): Data type of the tensor (e.g., ttnn.bfloat16 or ttnn.float32).
            Default is ttnn.bfloat16.
        layout (ttnn.Layout, optional): Tensor layout (e.g., ROW_MAJOR_LAYOUT or TILE_LAYOUT).
            Default is ROW_MAJOR_LAYOUT.
        memory_config (ttnn.MemoryConfig, optional): Device memory configuration to use
            (e.g., DRAM_MEMORY_CONFIG or L1_MEMORY_CONFIG). Default is DRAM.

    Returns:
        ttnn.Tensor:
            A tensor of shape (step_x, step_y, 4) containing the precomputed bilinear weights,
            stored on the given device with the specified dtype, layout, and memory configuration.
    """

    # Coordinate grid in [0,1]
    x = torch.arange(0, step_x, dtype=torch.float32)
    y = torch.arange(0, step_y, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    grid_x = grid_x / (step_x - 1)
    grid_y = grid_y / (step_y - 1)

    # Bilinear weights (vectorized)
    dx, dy = grid_x, grid_y
    dxdy = dx * dy

    w1 = 1 - dx - dy + dxdy   # top-left
    w2 = dy - dxdy            # top-right
    w3 = dx - dxdy            # bottom-left
    w4 = dxdy                 # bottom-right

    weight_hash = torch.stack((w1, w2, w3, w4), dim=-1).reshape(1, -1)  # (1, step_x * step_y * 4)

    # Convert to TTNN tensor
    weight_hash_tt = ttnn.from_torch(
        weight_hash,
        device=device,
        dtype=dtype,
        layout=layout,
        memory_config=memory_config,
    )
    weight_hash_tt = ttnn.reallocate(weight_hash_tt)

    return weight_hash_tt

DEFAULT_WEIGHT_HASH_CONFIG = {
    "step_x": 100,
    "step_y": 100,
    "dtype": ttnn.bfloat16,
    "layout": ttnn.ROW_MAJOR_LAYOUT,
    "memory_config": ttnn.L1_MEMORY_CONFIG,
}

@ttnn.register_python_operation(
    name="ttnn.BilinearWeightHashConfig"
)
def BilinearWeightHashConfig(step_x=1_000, step_y=1_000, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """
    Create a configuration dictionary for constructing a bilinear weight hash table.

    This is a convenience wrapper to standardize the parameters required
    for building a bilinear interpolation lookup table (via
    `ttnn.bos_create_bilinear_hash`). It allows easy reuse across test
    cases and function calls.

    Args:
        step_x (int, optional):
            Number of discretization steps along the x-axis.
            Defines the resolution of the hash grid. Default is 100.
        step_y (int, optional):
            Number of discretization steps along the y-axis.
            Defines the resolution of the hash grid. Default is 100.
        dtype (ttnn.dtype, optional):
            Data type to use for the hash table (e.g., ttnn.bfloat16 or ttnn.float32).
            Default is ttnn.bfloat16.
        layout (ttnn.Layout, optional):
            Memory layout of the tensor (ROW_MAJOR_LAYOUT or TILE_LAYOUT).
            Default is ROW_MAJOR_LAYOUT.
        memory_config (ttnn.MemoryConfig, optional):
            Memory configuration specifying where the hash table is allocated
            (e.g., DRAM_MEMORY_CONFIG or L1_MEMORY_CONFIG).
            Default is DRAM.

    Returns:
        dict:
            A dictionary containing the configuration fields:
            {
                "step_x": int,
                "step_y": int,
                "dtype": ttnn.dtype,
                "layout": ttnn.Layout,
                "memory_config": ttnn.MemoryConfig,
            }

    Example:
        >>> config = ttnn.BilinearWeightHashConfig(step_x=256, step_y=256, memory_config=ttnn.L1_MEMORY_CONFIG)
        >>> bilinear_hash = ttnn.bos_create_bilinear_hash(device, **config)
    """
    return {
        "step_x": step_x,
        "step_y": step_y,
        "dtype": dtype,
        "layout": layout,
        "memory_config": memory_config,
    }