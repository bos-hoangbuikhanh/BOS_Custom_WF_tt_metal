import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class QwenTTVisionMLP(LightweightModule):
    def __init__(
        self,
        mesh_device,
        args,
        state_dict,
        weight_cache_path,
        dtype,
        state_dict_prefix=None,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.args = args
        self.state_dict = state_dict
        self.dim = args.dim

        def get_weight(name):
            return torch.transpose(state_dict[f"{state_dict_prefix}{name}.weight"], -2, -1)

        def get_bias(name):
            return state_dict[f"{state_dict_prefix}{name}.bias"]

        def cache_name(name):
            if args.dummy_weights:
                return None
            return weight_cache_path / f"{state_dict_prefix}{name}"

        def as_tensor(name, dtype, is_bias=False):
            tensor_data = get_bias(name) if is_bias else get_weight(name)
            weight_name = f"{name}.{'bias' if is_bias else 'weight'}"
            return ttnn.as_tensor(
                tensor_data,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_name(weight_name),
            )

        self.linear_fc1_weight = as_tensor("linear_fc1", dtype)
        self.linear_fc1_bias = as_tensor("linear_fc1", dtype, is_bias=True)

        self.linear_fc2_weight = as_tensor("linear_fc2", dtype)
        self.linear_fc2_bias = as_tensor("linear_fc2", dtype, is_bias=True)

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            dst_full_sync_en=False,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        fc1_out = ttnn.linear(
            x,
            self.linear_fc1_weight,
            bias=self.linear_fc1_bias,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        act_out = ttnn.gelu(fc1_out)
        ttnn.deallocate(fc1_out)

        if self.args.num_devices > 1:
            act_out = ttnn.all_gather(act_out, dim=3, num_links=1)

        fc2_out = ttnn.linear(
            act_out,
            self.linear_fc2_weight,
            bias=self.linear_fc2_bias,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(act_out)

        if self.args.num_devices > 1:
            fc2_out = ttnn.all_gather(fc2_out, dim=len(fc2_out.shape) - 1, num_links=1)

        return fc2_out
