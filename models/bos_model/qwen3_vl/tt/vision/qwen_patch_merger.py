"""
This is the patch merger implementation for the Qwen3-VL Vision Model.
It can be used for both the final merger and the deepstack mergers.

It implements logic based on the `use_postshuffle_norm` flag:
- False (final merger):  output = MLP(reshape(norm(x)))
- True (deepstack):     output = MLP(norm(reshape(x)))
"""

import torch

import ttnn
from models.bos_model.qwen3_vl.tt.vision.qwen_layernorm import TtVisionLayerNorm
from models.common.lightweightmodule import LightweightModule


class TTQwen3_VLPatchMerger(LightweightModule):
    def __init__(
        self,
        mesh_device,
        args,
        state_dict,
        weight_cache_path,
        dtype,
        state_dict_prefix=None,
        use_postshuffle_norm=False,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.args = args
        self.use_postshuffle_norm = use_postshuffle_norm

        pre_norm_dim = args.vision_dim
        self.spatial_merge_size = args.spatial_merge_size
        self.post_norm_dim = pre_norm_dim * (self.spatial_merge_size**2)
        self.eps = 1e-6

        if self.use_postshuffle_norm:
            # Deepstack: Norms *after* reshape (dim 4096)
            self.norm_input_dim = self.post_norm_dim
        else:
            # Final Merger: Norms *before* reshape (dim 1024)
            self.norm_input_dim = pre_norm_dim

        def get_weight(name):
            return torch.transpose(state_dict[f"{state_dict_prefix}{name}.weight"], -2, -1)

        def get_bias(name):
            return state_dict[f"{state_dict_prefix}{name}.bias"]

        def as_tensor(tensor_data, name, dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.as_tensor(
                tensor_data,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
                layout=layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.norm = TtVisionLayerNorm(
            device=mesh_device,
            dim=self.norm_input_dim,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}norm.",
            dtype=dtype,
            eps=self.eps,
            weight_cache_path=weight_cache_path,
        )

        self.linear_fc1_weight = as_tensor(get_weight("linear_fc1"), "linear_fc1.weight", dtype)
        self.linear_fc1_bias = as_tensor(get_bias("linear_fc1"), "linear_fc1.bias", dtype)

        self.linear_fc2_weight = as_tensor(get_weight("linear_fc2"), "linear_fc2.weight", dtype)
        self.linear_fc2_bias = as_tensor(get_bias("linear_fc2"), "linear_fc2.bias", dtype)

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            dst_full_sync_en=False,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self.use_postshuffle_norm:
            # Deepstack: Reshape -> Norm
            x_reshaped = ttnn.reshape(x, (1, 1, -1, self.post_norm_dim))
            x_norm = self.norm(x_reshaped)
            ttnn.deallocate(x_reshaped)
        else:
            # Final Merger: Norm -> Reshape
            x_norm_pre = self.norm(x)
            x_norm = ttnn.reshape(x_norm_pre, (1, 1, -1, self.post_norm_dim))
            ttnn.deallocate(x_norm_pre)

        fc1_out = ttnn.linear(
            x_norm,
            self.linear_fc1_weight,
            bias=self.linear_fc1_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(x_norm)

        if self.args.num_devices > 1:
            fc1_out = ttnn.all_gather(fc1_out, dim=3)

        act_out = ttnn.gelu(fc1_out)
        ttnn.deallocate(fc1_out)

        fc2_out = ttnn.linear(
            act_out,
            self.linear_fc2_weight,
            bias=self.linear_fc2_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(act_out)

        if self.args.num_devices > 1:
            fc2_out = ttnn.all_gather(fc2_out, dim=3, num_links=1)

        output = ttnn.clone(ttnn.reshape(fc2_out, (-1, fc2_out.shape[-1])))
        ttnn.deallocate(fc2_out)

        return output
