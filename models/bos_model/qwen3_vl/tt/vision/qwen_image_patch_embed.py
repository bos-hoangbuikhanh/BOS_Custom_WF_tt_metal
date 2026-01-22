import ttnn


class TTQwen3_VisionPatchEmbed:
    def __init__(
        self,
        device,
        args,
        patch_size,
        temporal_patch_size,
        in_channels,
        embed_dim,
        state_dict,
        weight_key,
        layer_num=None,
        state_dict_prefix="",
        weight_cache_path=None,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        weight_dtype=ttnn.bfloat16,
        mode="decode",
    ):
        super().__init__()
        self.mode = mode
        self.device = device
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.weight_memory_config = weight_memory_config
        self.weight_dtype = weight_dtype
        self.args = args

        weight_name = f"{weight_key}proj.weight"
        bias_name = f"{weight_key}proj.bias"

        torch_weight = state_dict[weight_name]
        torch_bias = state_dict[bias_name]

        # Flatten the kernel: [C_out, C_in, T_k, H_k, W_k] -> [C_out, C_in * T_k * H_k * W_k]
        weight_matrix = torch_weight.view(self.embed_dim, -1)

        cache_name = lambda name: weight_cache_path / (f"{name}")

        # Transpose for matmul: [C_in * T_k * H_k * W_k, C_out]
        self.weight = ttnn.as_tensor(
            weight_matrix.T,
            device=self.device,
            dtype=self.weight_dtype,
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=-1),
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.weight_memory_config,
            cache_file_name=cache_name(weight_name),
        )

        self.bias = ttnn.as_tensor(
            torch_bias,
            device=self.device,
            dtype=self.weight_dtype,
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=-1),
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.weight_memory_config,
            cache_file_name=cache_name(bias_name),
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            dst_full_sync_en=False,
        )

    def __call__(self, x):
        # Assume input x is [1, 1, NumPatches, C*T_k*H_k*W_k]
        # Reshape to [NumPatches, C*T_k*H_k*W_k]
        x_flattened = ttnn.reshape(x, (x.shape[2], -1))

        output = ttnn.linear(x_flattened, self.weight, bias=self.bias, compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(x_flattened)

        if self.args.num_devices > 1:
            output = ttnn.all_gather(output, dim=1, num_links=1)

        return output
