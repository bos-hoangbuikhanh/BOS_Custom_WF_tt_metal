import ttnn
from models.common.lightweightmodule import LightweightModule


class TtVisionLayerNorm(LightweightModule):
    def __init__(self, device, dim, state_dict, state_dict_prefix, dtype, eps: float = 1e-6, weight_cache_path=None):
        super().__init__()
        self.device = device
        self.eps = eps

        weight_name = f"{state_dict_prefix}weight"
        bias_name = f"{state_dict_prefix}bias"

        weight_data = state_dict[weight_name]
        bias_data = state_dict[bias_name]

        cache_name = lambda name: weight_cache_path / (f"{name}")

        self.weight = ttnn.as_tensor(
            weight_data,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(weight_name),
        )
        self.bias = ttnn.as_tensor(
            bias_data,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(bias_name),
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x_norm = ttnn.layer_norm(
            x,
            epsilon=self.eps,
            weight=self.weight,
            bias=self.bias,
        )

        return x_norm
