from loguru import logger

import ttnn
from models.bos_model.qwen3_vl.tt.model import Transformer
from models.bos_model.qwen3_vl.tt.vision.qwen_vision_model import TtQwen3_VisionTransformerPretrainedModel


class TtQwen_Model(Transformer):
    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    ):
        super().__init__(
            args,
            dtype,
            mesh_device,
            state_dict,
            weight_cache_path,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
        )

        self.vision_model = TtQwen3_VisionTransformerPretrainedModel(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix="visual.",
            dtype=dtype,
            model_args=args,
            weight_cache_path=args.weight_cache_path(dtype),
            layers=args.vision_n_layers,
        )

        self.deepstack_feature_lists = None
        self.visual_pos_masks_torch = None

    def prepare_inputs_prefill(self, pt_tokens, start_pos=0, page_table=None, chunk_page_table=None, **kwargs):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device.
        """
        S = pt_tokens.shape[-1]
        tokens = ttnn.from_torch(
            pt_tokens.reshape(1, 1, 1, -1),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        tokens_embd = self.embd(tokens)

        # ✅ Check if image-related inputs exist
        pixel_values = kwargs.get("pixel_values", None)

        if pixel_values is not None:
            vision_output, deepstack_outputs = self.compute_vision_token(**kwargs)

            self.deepstack_feature_lists = deepstack_outputs

            tokens_embd = ttnn.to_torch(tokens_embd, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1))
            comp_vision_output = ttnn.to_torch(
                vision_output, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
            )[: vision_output.shape[0], :]

            ttnn.deallocate(vision_output)

            image_features = comp_vision_output.squeeze(0)

            # Create mask for scattering main features
            # Note: 151655 is the <image_token_id>
            special_image_mask = (pt_tokens == 151655).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(tokens_embd)

            # Scatter image features into text embeddings
            image_features = image_features.to(tokens_embd.device, tokens_embd.dtype)
            tokens_embd = tokens_embd.masked_scatter(special_image_mask, image_features)

            # Create and store the boolean mask for deepstack injection ---
            # The ref model uses `visual_pos_masks = image_mask | video_mask`
            # For now, we only handle images.
            image_mask_bool = pt_tokens == 151655  # Shape [1, Seq_len]
            self.visual_pos_masks_torch = image_mask_bool.squeeze(0)  # Shape [Seq_len]

        else:
            tokens_embd = ttnn.to_torch(tokens_embd, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1))

        tokens_embd = self.args.prepare_residual_tensor_prefill(
            tokens_embd,
        )

        tokens_embd = ttnn.unsqueeze_to_4D(tokens_embd)

        # Slice the rot mats to the prefill seqlen
        assert (
            self.rope_setup.cos_matrix.shape[2] >= start_pos + S
        ), f"Padded prefill end idx {start_pos + S} exceeds max seq len {self.rope_setup.cos_matrix.shape[2]}"

        tt_rot_mats_prefill_global = [
            self.rope_setup.cos_matrix[:, :, start_pos : start_pos + S, :],
            self.rope_setup.sin_matrix[:, :, start_pos : start_pos + S, :],
        ]

        if page_table is not None:
            tt_page_table = ttnn.from_torch(
                page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_page_table = None

        if chunk_page_table is not None:
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_chunk_page_table = None

        return tokens_embd, tt_rot_mats_prefill_global, tt_page_table, tt_chunk_page_table

    def compute_vision_token(self, pixel_values, image_grid_thw):
        pixel_values = self.args.prepare_residual_tensor_prefill(pixel_values.unsqueeze(0), force_replicated=True)

        vision_output, deepstack_outputs = self.vision_model(pixel_values, image_grid_thw)
        return vision_output, deepstack_outputs

    def _deepstack_process_ttnn(self, hidden_states, visual_pos_mask, visual_embeds):
        """
        Performs the deepstack injection on device.
        This is a ttnn port of:
         `hidden_states[visual_pos_masks, :].clone() + visual_embeds`

        hidden_states: [1, 1, S, D] ttnn.Tensor (TILE)
        visual_pos_mask: [S] torch.bool Tensor (on host)
        visual_embeds: [NumVisionTokens, D] ttnn.Tensor (TILE)
        """

        hidden_states_torch = ttnn.to_torch(
            hidden_states, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1)
        )
        hidden_states_torch = hidden_states_torch.squeeze(0).squeeze(0)  # [S, D]

        visual_embeds_torch = ttnn.to_torch(
            visual_embeds, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        )
        visual_embeds_torch = visual_embeds_torch[: visual_embeds.shape[0], :]  # [NumVisionTokens, D]

        hidden_states_torch[visual_pos_mask, :] = hidden_states_torch[
            visual_pos_mask, :
        ].clone() + visual_embeds_torch.to(hidden_states_torch.dtype)

        hidden_states_torch = hidden_states_torch.reshape(1, 1, *hidden_states_torch.shape)  # [1, 1, S, D]

        ttnn.deallocate(hidden_states)

        new_hidden_states = ttnn.from_torch(
            hidden_states_torch,
            device=self.mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        return new_hidden_states

    def forward(
        self,
        x,
        current_pos,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
        **kwargs,
    ):
        deepstack_visual_embeds = getattr(self, "deepstack_feature_lists", None)

        visual_pos_masks = getattr(self, "visual_pos_masks_torch", None)

        for i, layer in enumerate(self.layers):
            if (
                mode == "prefill"
                and deepstack_visual_embeds is not None
                and visual_pos_masks is not None
                and i in self.args.deepstack_visual_indexes
            ):
                logger.info(f"Injecting deepstack features at text layer {i}...")
                merger_idx = self.args.deepstack_visual_indexes.index(i)
                deepstack_feature = deepstack_visual_embeds[merger_idx]  # [S_vision_merged, D_text]

                # Perform the injection
                x = self._deepstack_process_ttnn(x, visual_pos_masks, deepstack_feature)

                # Deallocate the feature tensor after use
                ttnn.deallocate(deepstack_feature)

            # Standard text layer execution
            x = layer(
                x,
                current_pos,
                rot_mats,
                user_id,
                mode,
                page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache[i] if kv_cache is not None else None,
            )

        if mode == "prefill" and get_last_token == -1:
            return x

        if get_last_token != -1:
            x = ttnn.slice(x, (0, 0, get_last_token, 0), (1, 1, get_last_token + 32, x.shape[-1]))

        x = self.norm(x, mode=mode)

        if mode == "prefill" and self.model_config["LM_HEAD_INPUT_MEMCFG"].is_sharded():
            x = ttnn.interleaved_to_sharded(x, self.model_config["LM_HEAD_INPUT_MEMCFG"])

        x = self.lm_head(x)

        if mode == "prefill":
            x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
            x = ttnn.to_memory_config(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return x
