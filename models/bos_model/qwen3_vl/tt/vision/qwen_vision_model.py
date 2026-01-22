"""
This is the end-to-end architecture of the Qwen-VL 3 vision model.

This file has been refactored to *exactly* match the Qwen3VLVisionModel
reference implementation to fix PCC discrepancies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

import ttnn
from models.bos_model.qwen3_vl.tt.vision.qwen_image_block import TtQwen3_VLVisionBlock
from models.bos_model.qwen3_vl.tt.vision.qwen_image_patch_embed import TTQwen3_VisionPatchEmbed
from models.bos_model.qwen3_vl.tt.vision.qwen_patch_merger import TTQwen3_VLPatchMerger
from models.bos_model.qwen3_vl.tt.vision.qwen_rope import TTQwen3_VisionRotaryEmbedding
from models.common.lightweightmodule import LightweightModule


class TtQwen3_VisionTransformerPretrainedModel(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        model_args,
        layers,
        block_key="",
        gated=False,
    ):
        super().__init__()
        self.spatial_merge_size = model_args.spatial_merge_size
        self.patch_size = model_args.vision_patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.mesh_device = mesh_device
        self.config = model_args.hf_config.vision_config

        hidden_size = model_args.vision_dim
        n_heads = model_args.vision_attn_n_heads
        temporal_patch_size = model_args.temporal_patch_size

        self.patch_embed = TTQwen3_VisionPatchEmbed(
            device=mesh_device,
            args=model_args,
            patch_size=self.patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=3,
            embed_dim=hidden_size,
            state_dict=state_dict,
            weight_key=f"{state_dict_prefix}patch_embed.",
            layer_num=None,
            state_dict_prefix="",
            weight_cache_path=weight_cache_path,
            weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            weight_dtype=dtype,
        )

        self.pos_embed = nn.Embedding.from_pretrained(state_dict[f"{state_dict_prefix}pos_embed.weight"])
        self.num_grid_per_side = int(model_args.hf_config.vision_config.num_position_embeddings**0.5)

        head_dim = hidden_size // n_heads

        self.rotary_pos_emb = TTQwen3_VisionRotaryEmbedding(
            device=mesh_device,
            dim=head_dim // 2,
            theta=10000.0,
        )

        self.blocks = [
            TtQwen3_VLVisionBlock(
                mesh_device=mesh_device,
                state_dict=state_dict,
                state_dict_prefix=f"{state_dict_prefix}blocks.{i}.",
                weight_cache_path=weight_cache_path,
                dtype=dtype,
                model_args=model_args,
            )
            for i in tqdm(range(layers), desc=f"Loading vision transformer blocks")
        ]

        self.merger = TTQwen3_VLPatchMerger(
            mesh_device=mesh_device,
            args=model_args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            dtype=dtype,
            state_dict_prefix=f"{state_dict_prefix}merger.",
            use_postshuffle_norm=False,
        )

        self.deepstack_visual_indexes = model_args.deepstack_visual_indexes
        self.deepstack_merger_list = [
            TTQwen3_VLPatchMerger(
                mesh_device=mesh_device,
                args=model_args,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                dtype=dtype,
                state_dict_prefix=f"{state_dict_prefix}deepstack_merger_list.{i}.",
                use_postshuffle_norm=True,
            )
            for i in tqdm(range(len(self.deepstack_visual_indexes)), desc="Loading deepstack mergers")
        ]

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size

        max_hw = int(grid_thw[:, 1:].max().item())

        freq_table_tt = self.rotary_pos_emb(max_hw)

        freq_table = ttnn.to_torch(freq_table_tt, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))

        ttnn.deallocate(freq_table_tt)

        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)  # block row indices
            block_cols = torch.arange(merged_w, device=device)  # block col indices
            intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
            intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def forward(self, hidden_states, grid_thw):
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)

        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        hidden_states_torch = ttnn.to_torch(hidden_states)

        hidden_states_torch = hidden_states_torch + pos_embeds.to(hidden_states_torch.device)

        hidden_states = ttnn.from_torch(
            hidden_states_torch,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len = rotary_pos_emb.shape[0]

        hidden_states = ttnn.reshape(hidden_states, [1, 1, seq_len, -1])

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos_tensor = ttnn.from_torch(
            emb.cos(),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            layout=ttnn.TILE_LAYOUT,
        )
        sin_tensor = ttnn.from_torch(
            emb.sin(),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            layout=ttnn.TILE_LAYOUT,
        )
        position_embeddings = (cos_tensor, sin_tensor)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []

        for layer_num, blk in enumerate(tqdm(self.blocks, desc="Running Vision Blocks")):
            ttnn.ReadDeviceProfiler(self.mesh_device)
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)

            if layer_num in self.deepstack_visual_indexes:
                merger_idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.deepstack_merger_list[merger_idx](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        ttnn.deallocate(cos_tensor)
        ttnn.deallocate(sin_tensor)

        hidden_states = self.merger(hidden_states)

        logger.info("...Vision pipeline executed successfully...")

        return hidden_states, deepstack_feature_lists
