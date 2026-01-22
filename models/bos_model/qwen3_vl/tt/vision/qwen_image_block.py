"""
This is the vision block used in the Qwen-VL-7B architecture
consisting of RMSnorm and self-attention layer followed by an MLP layer.
"""

import ttnn
from models.bos_model.qwen3_vl.tt.vision.qwen_image_attention import TtQwen3_VLVisionSdpaAttention
from models.bos_model.qwen3_vl.tt.vision.qwen_image_mlp import QwenTTVisionMLP
from models.bos_model.qwen3_vl.tt.vision.qwen_layernorm import TtVisionLayerNorm
from models.common.lightweightmodule import LightweightModule


class TtQwen3_VLVisionBlock(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        dtype,
        model_args,
        weight_cache_path=None,
        state_dict_prefix=None,
    ):
        super().__init__()

        self.norm1 = TtVisionLayerNorm(
            device=mesh_device,
            dim=model_args.vision_dim,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}norm1.",  # Note the final dot
            dtype=dtype,
            weight_cache_path=weight_cache_path,
        )

        self.norm2 = TtVisionLayerNorm(
            device=mesh_device,
            dim=model_args.vision_dim,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}norm2.",  # Note the final dot
            dtype=dtype,
            weight_cache_path=weight_cache_path,
        )

        self.attn = TtQwen3_VLVisionSdpaAttention(
            mesh_device,
            state_dict,
            state_dict_prefix=f"{state_dict_prefix}attn.",
            weight_cache_path=model_args.weight_cache_path(dtype),
            dtype=dtype,
            configuration=model_args,
        )

        self.mlp = QwenTTVisionMLP(
            mesh_device=mesh_device,
            args=model_args,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}feed_forward.",
            weight_cache_path=model_args.weight_cache_path(dtype),
            dtype=dtype,
        )

    def forward(self, hidden_states, cu_seqlens, position_embeddings):
        hidden_states = ttnn.add(
            hidden_states,
            self.attn(
                self.norm1(hidden_states),
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            ),
        )

        hidden_states = ttnn.add(hidden_states, self.mlp(self.norm2(hidden_states)))

        return hidden_states
