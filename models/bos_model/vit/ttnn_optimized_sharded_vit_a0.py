# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import inspect

import torch
import transformers
from ttnn.dot_access import DotAccessDict
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight, preprocess_model_parameters

import ttnn
from ttnn.core import divup


def update_model_config(config, batch_size):
    should_reallocate_in_attention = False
    core_grid = ttnn.CoreGrid(x=5, y=4)

    TILE_HEIGHT = 32

    patch_count = config.image_size // config.patch_size  # 224/16=14
    seqL = patch_count * patch_count  # 196
    seqL_padded = divup(seqL, TILE_HEIGHT) * TILE_HEIGHT  # 224
    seqL_t = seqL_padded // TILE_HEIGHT  # 224 / 32 = 7
    dim_t = config.hidden_size // TILE_HEIGHT  # 768 / 32 = 24
    dim_t__y = dim_t // core_grid.y  # 6
    head_num = config.num_attention_heads  # 12
    head_seqL_t__y = (head_num * seqL_t) // core_grid.y  # 21
    head_size_t = dim_t // head_num  # 2
    # 1000 classes padded to 1152
    class__y = (1152 // TILE_HEIGHT) // core_grid.y  #   9
    class_subb_w = class__y
    if class_subb_w > 8:  # max ratio of sub_block_w / sub_block_h = 8
        if class_subb_w % 3 == 0:
            class_subb_w = class__y // 3
        elif class_subb_w % 2 == 0:
            class_subb_w = class__y // 2
        else:
            class_subb_w = 1

    # sharding configs
    program_configs = {
        "layernorm_before_program_config": ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            subblock_w=3,
            block_h=seqL_t,
            block_w=dim_t__y,
            inplace=False,
        ),
        "query_key_value_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=dim_t__y,
            out_subblock_h=1,
            out_subblock_w=dim_t__y,
            per_core_M=seqL_t,
            per_core_N=3 * dim_t__y,
            transpose_mcast=True,
            fused_activation=None,
        ),
        "query_by_key_matmul_program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=head_size_t,
            out_subblock_h=1,
            out_subblock_w=seqL_t,
            per_core_M=21,
            per_core_N=head_seqL_t__y // 3,
        ),
        "softmax_program_config": ttnn.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            subblock_w=seqL_t,
            block_h=head_seqL_t__y,
            block_w=seqL_t,
        ),
        "attention_probabilities_by_value_matmul_program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=seqL_t,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=21,
            per_core_N=head_size_t,
        ),
        "self_output_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=dim_t__y,
            out_subblock_h=1,
            out_subblock_w=dim_t__y,
            per_core_M=seqL_t,
            per_core_N=dim_t__y,
            transpose_mcast=True,
            fused_activation=None,
        ),
        "layernorm_after_output_program_config": ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            subblock_w=dim_t__y,
            block_h=seqL_t,
            block_w=dim_t__y,
            inplace=False,
        ),
        "ff1_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=3,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=seqL_t,
            per_core_N=dim_t__y * 4,
            transpose_mcast=True,
            fused_activation=(ttnn.UnaryOpType.GELU, True),
        ),
        "ff2_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=dim_t__y,
            per_core_M=seqL_t,
            per_core_N=dim_t__y,
            transpose_mcast=True,
            fused_activation=None,
        ),
        "classifer_matmul_program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=dim_t__y,
            out_subblock_h=1,
            out_subblock_w=class_subb_w,
            per_core_M=seqL_t,
            per_core_N=class__y,
            transpose_mcast=True,
            fused_activation=None,
        ),
        "ln_compute_config": ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        ),
    }

    # properties are not in the output of config.to_dict() but can be used later in the model
    # e.g. https://github.com/huggingface/transformers/blob/v4.53.0/src/transformers/configuration_utils.py#L368-L378
    property_names = [name for name, value in inspect.getmembers(config.__class__) if isinstance(value, property)]
    properties = {name: getattr(config, name) for name in property_names}

    return DotAccessDict(
        dict(
            **(config.to_dict() | properties),
            core_grid=core_grid,
            should_reallocate_in_attention=should_reallocate_in_attention,
            program_configs=program_configs,
        )
    )


# https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/vit/modeling_vit.py


def vit_patch_embeddings(config, pixel_values, *, parameters, unittest_check=False):
    batch_size, img_h, img_w, img_c = pixel_values.shape  # permuted input NHWC
    patch_size = config.patch_size
    patch_count = img_h // patch_size  # 14
    patch_size_sq_trpl = int(patch_size * patch_size * 3)  # 768
    patch_count_all = int(patch_count * patch_count)  # 196
    stride_h = patch_size
    stride_w = 1

    folded_pixel_values = ttnn.fold(pixel_values, stride_h, stride_w)  # 1568, 1024
    ttnn.deallocate(pixel_values)
    folded_pixel_values = ttnn.to_memory_config(folded_pixel_values, memory_config=ttnn.L1_MEMORY_CONFIG)
    # Convert back to interleaved or otherwise to_layout will fail
    folded_pixel_values = ttnn.to_layout(folded_pixel_values, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)

    if unittest_check:
        parameters = parameters.vit.embeddings.patch_embeddings

    patch_embedding_output = ttnn.linear(
        folded_pixel_values,
        parameters.projection.weight,
        bias=parameters.projection.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=config.core_grid,
    )

    patch_embedding_output = ttnn.to_layout(patch_embedding_output, layout=ttnn.ROW_MAJOR_LAYOUT)
    patch_embedding_output = ttnn.reshape(patch_embedding_output, (batch_size, patch_count_all, patch_size_sq_trpl))

    return patch_embedding_output


def vit_embeddings(
    config,
    pixel_values,
    cls_token,
    position_embeddings,
    *,
    parameters,
):
    parameters = parameters.vit.embeddings

    l1_memory_config = ttnn.L1_MEMORY_CONFIG

    patch_embeddings = vit_patch_embeddings(config, pixel_values, parameters=parameters.patch_embeddings)
    embedding_output = ttnn.concat([cls_token, patch_embeddings], -2, memory_config=l1_memory_config)
    embedding_output = ttnn.to_layout(embedding_output, layout=ttnn.TILE_LAYOUT)
    embedding_output = ttnn.add(
        embedding_output, position_embeddings, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
    )

    return embedding_output


def vit_attention(
    config,
    hidden_states,
    parameters,
):
    num_heads = config.num_attention_heads  # num_heads = 12
    *_, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    query_key_value = ttnn.linear(
        hidden_states,
        parameters.attention.query_key_value.weight,
        bias=parameters.attention.query_key_value.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["query_key_value_matmul_program_config"],
    )

    (
        query,
        key,
        value,
    ) = ttnn.transformer.split_query_key_value_and_split_heads(
        query_key_value,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        num_heads=num_heads,
    )

    ttnn.deallocate(query_key_value)
    ttnn.deallocate(hidden_states)
    if config.should_reallocate_in_attention:
        value = ttnn.reallocate(value)

    attention_scores = ttnn.matmul(
        query,
        key,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["query_by_key_matmul_program_config"],
    )
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    scale = 1.0 / (head_size**0.5)
    attention_scores = ttnn.mul_(
        attention_scores,
        scale,
    )

    attention_probs = ttnn.softmax_in_place(
        attention_scores,
        program_config=config.program_configs["softmax_program_config"],
    )

    context_layer = ttnn.matmul(
        attention_probs,
        value,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["attention_probabilities_by_value_matmul_program_config"],
    )
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value)

    context_layer = ttnn.transformer.concatenate_heads(
        context_layer,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
    )

    self_output = ttnn.linear(
        context_layer,
        parameters.output.dense.weight,
        bias=parameters.output.dense.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["self_output_matmul_program_config"],
    )
    ttnn.deallocate(context_layer)
    if config.should_reallocate_in_attention:
        self_output = ttnn.reallocate(self_output)

    return self_output


def vit_intermediate(
    config,
    hidden_states,
    *,
    parameters,
):
    output = ttnn.linear(
        hidden_states,
        parameters.dense.weight,
        bias=parameters.dense.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["ff1_matmul_program_config"],
    )
    ttnn.deallocate(hidden_states)

    return output


def vit_output(
    config,
    hidden_states,
    residual,
    *,
    parameters,
):
    output = ttnn.linear(
        hidden_states,
        parameters.dense.weight,
        bias=parameters.dense.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["ff2_matmul_program_config"],
    )
    ttnn.deallocate(hidden_states)

    output = ttnn.add(output, residual, memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
    ttnn.deallocate(residual)

    return output


def vit_feedforward(
    config,
    hidden_states,
    attention_output,
    *,
    parameters,
):
    intermediate = vit_intermediate(config, hidden_states, parameters=parameters.intermediate)
    hidden_states = vit_output(config, intermediate, attention_output, parameters=parameters.output)
    return hidden_states


def vit_layer(
    config,
    hidden_states,
    parameters,
):
    layernorm_before_output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layernorm_before.weight,
        bias=parameters.layernorm_before.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        program_config=config.program_configs["layernorm_before_program_config"],
        compute_kernel_config=config.program_configs["ln_compute_config"],
    )

    multi_head_attention_output = vit_attention(
        config,
        layernorm_before_output,
        parameters=parameters.attention,
    )

    multi_head_attention_output = ttnn.add(
        multi_head_attention_output,
        hidden_states,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
    )

    layernorm_after_output = ttnn.layer_norm(
        multi_head_attention_output,
        weight=parameters.layernorm_after.weight,
        bias=parameters.layernorm_after.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        program_config=config.program_configs["layernorm_after_output_program_config"],
        compute_kernel_config=config.program_configs["ln_compute_config"],
    )

    feedforward_output = vit_feedforward(
        config,
        layernorm_after_output,
        multi_head_attention_output,
        parameters=parameters,
    )

    return feedforward_output


def vit_encoder(
    config,
    embeddings,
    parameters,
):
    TILE_HEIGHT = 32
    emb_N, emb_S, emb_D = embeddings.shape
    emb_S = divup(emb_S, TILE_HEIGHT) * TILE_HEIGHT
    encoder_input = ttnn.to_memory_config(
        embeddings,
        memory_config=ttnn.create_sharded_memory_config(
            [emb_N, emb_S, emb_D],
            core_grid=config.core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.COL_MAJOR,
        ),
        dtype=ttnn.bfloat8_b,
    )
    ttnn.deallocate(embeddings)

    for index, encoder_parameters in enumerate(parameters.layer):
        encoder_output = vit_layer(
            config,
            encoder_input,
            encoder_parameters,
        )
        encoder_input = encoder_output

    return encoder_output


def vit(
    config,
    pixel_values,
    cls_token,
    position_embeddings,
    parameters,
):
    embeddings_output = vit_embeddings(config, pixel_values, cls_token, position_embeddings, parameters=parameters)

    hidden_states = vit_encoder(
        config,
        embeddings_output,
        parameters=parameters.vit.encoder,
    )

    # Final LayerNorm
    output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.vit.layernorm.weight,
        bias=parameters.vit.layernorm.bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        program_config=config.program_configs["layernorm_before_program_config"],
    )

    # Classifier
    classifier_output = ttnn.linear(
        output,
        parameters.classifier.weight,
        bias=parameters.classifier.bias,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=config.program_configs["classifer_matmul_program_config"],
    )
    return classifier_output


def preprocess_inputs(
    input_ids,
    token_type_ids,
    device,
):
    batch_size, _ = input_ids.shape

    input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    token_type_ids = ttnn.from_torch(
        token_type_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    return input_ids, token_type_ids


def custom_preprocessor(torch_model, name):
    parameters = {}
    if isinstance(torch_model, transformers.models.vit.modeling_vit.ViTEmbeddings):
        weight = torch_model.patch_embeddings.projection.weight
        bias = torch_model.patch_embeddings.projection.bias

        three_times_hidden_size, c, _, _ = weight.shape
        pad_value = 4 - c
        preprocessed_weight = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, pad_value))
        preprocessed_weight = torch.permute(preprocessed_weight, (2, 3, 1, 0))
        preprocessed_weight = torch.reshape(
            preprocessed_weight, (int(three_times_hidden_size * (4 / c)), three_times_hidden_size)
        )

        parameters = {"patch_embeddings": {}}
        parameters["patch_embeddings"] = {"projection": {}}
        parameters["patch_embeddings"]["projection"]["weight"] = ttnn.from_torch(
            preprocessed_weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )
        parameters["patch_embeddings"]["projection"]["bias"] = ttnn.from_torch(
            bias.unsqueeze(0), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )

        parameters["cls_token"] = ttnn.from_torch(torch_model.cls_token, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        parameters["position_embeddings"] = ttnn.from_torch(
            torch_model.position_embeddings, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )

    if hasattr(torch_model, "query") and hasattr(torch_model, "key") and hasattr(torch_model, "value"):
        num_heads = 12
        head_size = 64
        hidden_size = num_heads * head_size * 3
        qkv_weight = torch.cat(
            [
                torch_model.query.weight.reshape([num_heads, head_size, -1]),
                torch_model.key.weight.reshape([num_heads, head_size, -1]),
                torch_model.value.weight.reshape([num_heads, head_size, -1]),
            ],
            dim=1,
        ).reshape([hidden_size, -1])
        qkv_bias = torch.cat(
            [
                torch_model.query.bias.reshape([num_heads, head_size]),
                torch_model.key.bias.reshape([num_heads, head_size]),
                torch_model.value.bias.reshape([num_heads, head_size]),
            ],
            dim=1,
        ).reshape([hidden_size])

        parameters = {"query_key_value": {}}
        parameters["query_key_value"]["weight"] = preprocess_linear_weight(qkv_weight, dtype=ttnn.bfloat8_b)
        parameters["query_key_value"]["bias"] = preprocess_linear_bias(qkv_bias, dtype=ttnn.bfloat8_b)

    elif isinstance(torch_model, torch.nn.Linear):
        # TODO: better way of detection for the classify linear weights
        if torch_model.weight.shape[0] == 1000:
            preprocessed_weight = torch.nn.functional.pad(torch_model.weight, (0, 0, 0, int(1152 - 1000)))
            preprocessed_bias = torch.nn.functional.pad(torch_model.bias, (0, int(1152 - 1000)))
            parameters["weight"] = preprocess_linear_weight(preprocessed_weight, dtype=ttnn.bfloat8_b)
            parameters["bias"] = preprocess_linear_bias(preprocessed_bias, dtype=ttnn.bfloat8_b)
        else:
            parameters["weight"] = preprocess_linear_weight(torch_model.weight, dtype=ttnn.bfloat8_b)
            parameters["bias"] = preprocess_linear_bias(torch_model.bias, dtype=ttnn.bfloat8_b)

    return parameters


class TtViT:
    def __init__(self, device, batch_size, torch_vit, use_trace=False, use_2cq=False):
        self.patch_size = 16
        self.batch_size = batch_size
        self.device = device
        self.parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_vit.to(torch.bfloat16),
            custom_preprocessor=custom_preprocessor,
            device=self.device,
        )
        self.config = update_model_config(torch_vit.config, self.batch_size)
        H, W, C = self.config.image_size, self.config.image_size // self.config.patch_size, self.config.patch_size * 4
        full_grid = self.device.compute_with_storage_grid_size()
        N_CORES = full_grid.x * full_grid.y
        self.input_l1_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.num_cores_to_corerangeset(N_CORES, full_grid),
                [batch_size * H * W // N_CORES, C],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        model_state_dict = torch_vit.state_dict()
        cls_token = model_state_dict["vit.embeddings.cls_token"]
        position_embeddings = model_state_dict["vit.embeddings.position_embeddings"]
        if self.batch_size > 1:
            cls_token = torch.nn.Parameter(cls_token.expand(batch_size, -1, -1))
            position_embeddings = torch.nn.Parameter(position_embeddings.expand(batch_size, -1, -1))
        else:
            cls_token = torch.nn.Parameter(cls_token)
            position_embeddings = torch.nn.Parameter(position_embeddings)

        self.cls_token = ttnn.from_torch(
            cls_token, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        self.position_embeddings = ttnn.from_torch(
            position_embeddings, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.head_masks = None

    def forward(self, pixel_values: torch.Tensor) -> ttnn.Tensor:
        """
        returns ttnn.Tensor on device
        """
        tt_pixel_values = self._prepare_inputs(pixel_values)
        tt_pixel_values = ttnn.to_device(
            tt_pixel_values,
            self.device,
            memory_config=self.input_l1_mem_config,
        )
        return self._vit_device_func(tt_pixel_values)

    def _prepare_inputs(self, pixel_values: torch.Tensor) -> ttnn.Tensor:
        """
        returns ttnn.Tensor on host with Shape([batch, 224, 14, 64])
        """
        pixel_values = torch.permute(pixel_values, (0, 2, 3, 1))
        pixel_values = torch.nn.functional.pad(pixel_values, (0, 1, 0, 0, 0, 0, 0, 0))
        batch_size, img_h, img_w, _ = pixel_values.shape  # permuted input NHWC
        assert batch_size == self.batch_size
        pixel_values = pixel_values.reshape(batch_size, img_h, img_w // self.patch_size, 4 * self.patch_size)
        pixel_values = ttnn.from_torch(
            pixel_values,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
        )
        return pixel_values

    def _vit_device_func(self, tt_pixel_values: ttnn.Tensor) -> ttnn.Tensor:
        return vit(self.config, tt_pixel_values, self.cls_token, self.position_embeddings, parameters=self.parameters)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
