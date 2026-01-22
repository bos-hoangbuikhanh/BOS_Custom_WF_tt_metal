import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


def rotate_half(x):
    """
    Applies rotary rotation, splitting on the last dimension.
    x.shape is [S, H, D]

    Uses ttnn.slice with EXCLUSIVE end indices.
    """
    s, h, d = x.shape

    # x1: Slices indices 0..31. End index is 32 (exclusive).
    x1 = ttnn.slice(x, (0, 0, 0), (s, h, d // 2))

    # x2: Slices indices 32..63. Start index is 32. End index is 64 (exclusive).
    x2 = ttnn.slice(x, (0, 0, d // 2), (s, h, d))

    return ttnn.concat([ttnn.neg(x2), x1], dim=-1)


def apply_rotary_pos_emb_vision_tt(q, k, cos, sin):
    """
    q, k: [Seq, Heads, Head_Dim]
    cos, sin: [Seq, Head_Dim]
    """
    # [Seq, Head_Dim] -> [Seq, 1, Head_Dim]
    cos = ttnn.unsqueeze(cos, 1)
    sin = ttnn.unsqueeze(sin, 1)

    # Broadcast multiply:
    # [S, H, D] * [S, 1, D]
    q_embed = ttnn.add(ttnn.mul(q, cos), ttnn.mul(rotate_half(q), sin))
    k_embed = ttnn.add(ttnn.mul(k, cos), ttnn.mul(rotate_half(k), sin))

    # Deallocate broadcasted tensors
    ttnn.deallocate(cos)
    ttnn.deallocate(sin)

    return q_embed, k_embed


class TtQwen3_VLVisionSdpaAttention(LightweightModule):
    def __init__(self, mesh_device, state_dict, state_dict_prefix, dtype, configuration, weight_cache_path=None):
        super().__init__()

        self.configuration = configuration
        self.mesh_device = mesh_device
        self.dtype = dtype

        self.hidden_size = configuration.vision_dim
        self.num_heads = configuration.vision_attn_n_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.scale = self.head_dim**-0.5

        self.padded_head_dim = math.ceil(self.head_dim / 32) * 32
        self.needs_padding = self.padded_head_dim != self.head_dim
        self.padding_amount = self.padded_head_dim - self.head_dim

        if self.needs_padding:
            print(f"Vision attention: head_dim={self.head_dim} requires padding to {self.padded_head_dim}")

        qkv_weight = state_dict[f"{state_dict_prefix}qkv.weight"]
        qkv_bias = state_dict[f"{state_dict_prefix}qkv.bias"]

        cache_name = lambda name: weight_cache_path / (f"{state_dict_prefix}{name}")

        self.qkv_weight = ttnn.as_tensor(
            torch.transpose(qkv_weight, -2, -1),
            device=mesh_device,
            dtype=dtype,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("qkv.weight"),
        )
        self.qkv_bias = ttnn.as_tensor(
            qkv_bias,
            device=mesh_device,
            dtype=dtype,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("qkv.bias"),
        )

        proj_weight = state_dict[f"{state_dict_prefix}proj.weight"]
        proj_bias = state_dict[f"{state_dict_prefix}proj.bias"]

        self.proj_weight = ttnn.as_tensor(
            torch.transpose(proj_weight, -2, -1),
            device=mesh_device,
            dtype=dtype,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("proj.weight"),
        )
        self.proj_bias = ttnn.as_tensor(
            proj_bias,
            device=mesh_device,
            dtype=dtype,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("proj.bias"),
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            dst_full_sync_en=False,
        )

    def forward(self, hidden_states, cu_seqlens, position_embeddings):
        """
        hidden_states: ttnn.Tensor of shape [1, 1, seq_len, hidden_size]
        position_embeddings: tuple (cos, sin) each of shape [seq_len, head_dim]
        """
        seq_len = hidden_states.shape[-2]
        cos, sin = position_embeddings

        qkv = ttnn.linear(
            hidden_states,
            self.qkv_weight,
            bias=self.qkv_bias,
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # shape [1, 1, seq_len, hidden_size*3]

        if self.configuration.num_devices > 1:
            qkv = ttnn.all_gather(qkv, dim=-1, num_links=1)

        # Reshape to [1, seq_len, 3, num_heads, head_dim]
        qkv_reshaped = ttnn.reshape(qkv, [1, seq_len, 3, self.num_heads, self.head_dim])
        ttnn.deallocate(qkv)

        # Permute to [3, 1, seq_len, num_heads, head_dim]
        qkv_permuted = ttnn.permute(qkv_reshaped, (2, 0, 1, 3, 4))
        ttnn.deallocate(qkv_reshaped)

        # Split along dim 0, with a chunk size of 1
        qkv_list = ttnn.split(qkv_permuted, split_size=1, dim=0)
        ttnn.deallocate(qkv_permuted)

        # Squeeze dims 0 and 1 to get [S, H, D] for RoPE
        q = ttnn.squeeze(qkv_list[0], 0)
        q = ttnn.squeeze(q, 0)

        k = ttnn.squeeze(qkv_list[1], 0)
        k = ttnn.squeeze(k, 0)

        v = ttnn.squeeze(qkv_list[2], 0)
        v = ttnn.squeeze(v, 0)

        q, k = apply_rotary_pos_emb_vision_tt(q, k, cos, sin)

        if self.needs_padding:
            # Current shape: [seq_len, num_heads, head_dim]
            # Pad the last dimension from head_dim to padded_head_dim
            q = ttnn.pad(q, [(0, 0), (0, 0), (0, self.padding_amount)], 0)
            k = ttnn.pad(k, [(0, 0), (0, 0), (0, self.padding_amount)], 0)
            v = ttnn.pad(v, [(0, 0), (0, 0), (0, self.padding_amount)], 0)

        # Reshape for SDPA: [Seq, Heads, Head_Dim] -> [1, Heads, Seq, Head_Dim]
        q = ttnn.permute(ttnn.unsqueeze(q, 0), (0, 2, 1, 3))
        k = ttnn.permute(ttnn.unsqueeze(k, 0), (0, 2, 1, 3))
        v = ttnn.permute(ttnn.unsqueeze(v, 0), (0, 2, 1, 3))

        attn_output = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False, scale=self.scale)

        attn_output = ttnn.reallocate(attn_output)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # Slice back to original head_dim if we padded
        if self.needs_padding:
            # attn_output shape: [1, num_heads, seq_len, padded_head_dim]
            # Slice to [1, num_heads, seq_len, head_dim]
            attn_output = ttnn.slice(
                attn_output,
                (0, 0, 0, 0),
                (attn_output.shape[0], attn_output.shape[1], attn_output.shape[2], self.head_dim),
            )

        # Reshape to [1, 1, seq_len, num_heads * head_dim] for linear
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, [1, 1, seq_len, self.hidden_size])

        output = ttnn.linear(
            attn_output,
            self.proj_weight,
            bias=self.proj_bias,
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_output)

        if self.configuration.num_devices > 1:
            output = ttnn.all_gather(output, dim=-1, num_links=1)

        # Reshape output to [Seq, Dim] to match reference
        output = ttnn.reshape(output, (seq_len, self.hidden_size))

        return output
