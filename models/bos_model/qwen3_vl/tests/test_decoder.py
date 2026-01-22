# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.bos_model.qwen3_vl.tt.common import PagedAttentionConfig, precompute_freqs
from models.bos_model.qwen3_vl.tt.decoder import TransformerBlock
from models.bos_model.qwen3_vl.tt.model_config import ModelArgs
from models.bos_model.qwen3_vl.tt.rope import RotarySetup
from models.common.utility_functions import comp_allclose, comp_pcc

# from models.common.utility_functions import skip_for_grayskull


@torch.no_grad()
# @skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "paged_attention",
    (
        True,
        # False
    ),
    ids=(
        "paged_attention",
        # "default_attention"
    ),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 32, "page_max_num_blocks": 1024}],
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (256,),  # For decode-only unit test, there's no need to run with large sequence lengths
)
def test_decoder_inference(
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    mesh_device,
    reset_seeds,
    ensure_gc,
):
    dtype = ttnn.bfloat8_b

    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len, cache_hf=True)
    # --- MODIFICATION START ---
    # model_args.n_layers = 1 # We no longer force 1 layer, we will use the default (36)
    # --- MODIFICATION END ---

    state_dict = model_args.load_state_dict()

    generation_start_pos = 0
    # --- MODIFICATION START ---
    # generation_length = 10 # This loop is being replaced by a layer loop
    # --- MODIFICATION END ---
    all_tests_pass = True

    # Setup RoPE transformation matrices
    rope_setup = RotarySetup(
        mesh_device,
        model_args.max_batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.rope_scaling,
    )
    transformation_mats = rope_setup.get_both_trans_mats()

    # Prepare page table for paged attention
    page_table_tt = None
    paged_attention_config = None

    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, -2) if (model_args.is_galaxy and batch_size > 1) else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

    seqlen = 1

    cos, sin = precompute_freqs(
        model_args.head_dim,
        model_args.max_seq_len * 2,
        model_args.rope_theta,
        model_args.rope_scaling.factor if model_args.rope_scaling else None,
        model_args.rope_scaling.original_max_position_embeddings if model_args.rope_scaling else None,
    )
    freqs_cis = torch.complex(cos, sin)

    # Initial positions
    current_pos = torch.tensor([generation_start_pos for _ in range(batch_size)])
    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    # --- MODIFICATION START ---
    # Loop over all layers instead of generation length
    for layer_num in range(model_args.n_layers):
        logger.info(f"[Decoder] Testing Layer {layer_num}")

        # --- MOVED LOGIC INSIDE LOOP ---
        # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
        first_layer_prefix = model_args.get_state_dict_prefix("TransformerBlock", layer_num)
        partial_state_dict = {
            k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
        }
        reference_model = model_args.reference_decoder()
        reference_model.load_state_dict(partial_state_dict)

        # Initialize TT model
        tt_model = TransformerBlock(
            args=model_args,
            mesh_device=mesh_device,
            dtype=dtype,
            state_dict=state_dict,
            layer_num=layer_num,
            weight_cache_path=model_args.weight_cache_path(dtype),
            transformation_mats=transformation_mats,
            paged_attention_config=paged_attention_config,
        )
        # --- END MOVED LOGIC ---

        # input = torch.randn(1, 32, 4096)
        pt_decode_input = (
            torch.rand(
                batch_size,
                seqlen,
                model_args.dim,
                # dtype=get_ref_model_dype(reference_model, model_args.model_name)
                dtype=torch.bfloat16,
            )
            * 2
        ) - 1
        tt_decode_input = pt_decode_input.clone()

        decode_input = model_args.prepare_residual_tensor_decode(
            tt_decode_input,
            # ttnn.DRAM_MEMORY_CONFIG,
            model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
        )

        # Get cos/sin matrices for the current position of each user
        rot_mats = rope_setup.get_rot_mats(current_pos)

        # Run TT model
        tt_out = tt_model(
            decode_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )
        tt_out = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
        )

        # Squeeze to match reference output shape [B, S, D] -> [S, D] since B=1
        tt_output_torch = (
            tt_out[:, 0:1, : model_args.max_batch_size, : model_args.dim]
            .view(-1, 1, model_args.dim)
            .squeeze(0)
            .squeeze(0)
        )

        # In this test all users have the same position
        freqs_cis_i = freqs_cis[current_pos[0], :].unsqueeze(0)

        # Reference model
        ref_output = reference_model(pt_decode_input, current_pos[0], freqs_cis_i, mask=None)

        # Squeeze ref output to match, as it has a batch dim [B, S, D] -> [S, D]
        ref_output = ref_output.squeeze(0)

        print(f"ttnn output: {tt_output_torch.shape}")
        print(f"Reference output: {ref_output.shape}")

        passing, pcc_message = comp_pcc(ref_output, tt_output_torch)

        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")

        if passing:
            logger.info(f"Decoder Block {layer_num} Passed!")
        else:
            logger.warning(f"Decoder Block {layer_num} Failed!")
            all_tests_pass = False

        # --- MODIFICATION START ---
        # We no longer increment position, as we are testing each layer independently at the same position
        # --- MODIFICATION END ---

    if all_tests_pass:
        logger.info(f"All {model_args.n_layers} decode layers Passed!")
    else:
        logger.warning("One or more layers of decode Failed!")
        assert all_tests_pass, f"PCC value is lower than {0.99} for some of the outputs. Check Warnings!"


# # SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# # SPDX-License-Identifier: Apache-2.0
# import os

# import pytest
# import torch
# from loguru import logger

# import ttnn
# from models.bos_model.qwen3_vl.tests.test_utils import get_ref_model_dype
# from models.bos_model.qwen3_vl.tt.common import PagedAttentionConfig, precompute_freqs
# from models.bos_model.qwen3_vl.tt.decoder import TransformerBlock
# from models.bos_model.qwen3_vl.tt.model_config import ModelArgs
# from models.bos_model.qwen3_vl.tt.rope import RotarySetup
# from models.common.utility_functions import comp_allclose, comp_pcc

# # from models.common.utility_functions import skip_for_grayskull


# @torch.no_grad()
# # @skip_for_grayskull("Requires wormhole_b0 to run")
# @pytest.mark.parametrize(
#     "mesh_device",
#     [
#         {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
#             os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
#         )
#     ],
#     indirect=True,
# )
# @pytest.mark.parametrize(
#     "paged_attention",
#     (
#         True,
#         # False
#     ),
#     ids=(
#         "paged_attention",
#         # "default_attention"
#     ),
# )
# @pytest.mark.parametrize(
#     "page_params",
#     [{"page_block_size": 32, "page_max_num_blocks": 1024}],
# )
# @pytest.mark.parametrize(
#     "batch_size",
#     (1,),
# )
# @pytest.mark.parametrize(
#     "max_seq_len",
#     (256,),  # For decode-only unit test, there's no need to run with large sequence lengths
# )
# def test_decoder_inference(
#     max_seq_len,
#     batch_size,
#     paged_attention,
#     page_params,
#     mesh_device,
#     reset_seeds,
#     ensure_gc,
# ):
#     dtype = ttnn.bfloat8_b

#     model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len, cache_hf=True)
#     model_args.n_layers = 1

#     state_dict = model_args.load_state_dict()

#     # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
#     first_layer_prefix = model_args.get_state_dict_prefix("TransformerBlock", 0)
#     partial_state_dict = {
#         k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
#     }
#     reference_model = model_args.reference_decoder()
#     reference_model.load_state_dict(partial_state_dict)

#     generation_start_pos = 0
#     generation_length = 10
#     all_tests_pass = True

#     # Setup RoPE transformation matrices
#     rope_setup = RotarySetup(
#         mesh_device,
#         model_args.max_batch_size,
#         model_args.head_dim,
#         model_args.max_seq_len,
#         model_args.rope_theta,
#         model_args.rope_scaling,
#     )
#     transformation_mats = rope_setup.get_both_trans_mats()

#     # Prepare page table for paged attention
#     page_table_tt = None
#     paged_attention_config = None

#     if paged_attention:
#         paged_attention_config = PagedAttentionConfig(
#             block_size=page_params["page_block_size"],
#             max_num_blocks=page_params["page_max_num_blocks"],
#         )
#         # Implied shuffling of blocks
#         permutation = torch.randperm(paged_attention_config.max_num_blocks)
#         # Page table which maps virtual blocks to physical
#         reverse_permutation = torch.argsort(permutation)
#         page_table = reverse_permutation.reshape(
#             model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
#         )
#         page_table_tt = ttnn.from_torch(
#             page_table,
#             device=mesh_device,
#             dtype=ttnn.int32,
#             layout=ttnn.ROW_MAJOR_LAYOUT,
#             mesh_mapper=ttnn.ShardTensor2dMesh(
#                 mesh_device,
#                 dims=(None, -2) if (model_args.is_galaxy and batch_size > 1) else (None, None),
#                 mesh_shape=model_args.cluster_shape,
#             ),
#         )

#     # Initialize TT model
#     tt_model = TransformerBlock(
#         args=model_args,
#         mesh_device=mesh_device,
#         dtype=dtype,
#         state_dict=state_dict,
#         layer_num=0,
#         weight_cache_path=model_args.weight_cache_path(dtype),
#         transformation_mats=transformation_mats,
#         paged_attention_config=paged_attention_config,
#     )

#     seqlen = 1

#     cos, sin = precompute_freqs(
#         model_args.head_dim,
#         model_args.max_seq_len * 2,
#         model_args.rope_theta,
#         model_args.rope_scaling.factor if model_args.rope_scaling else None,
#         model_args.rope_scaling.original_max_position_embeddings if model_args.rope_scaling else None,
#     )
#     freqs_cis = torch.complex(cos, sin)

#     # Initial positions
#     current_pos = torch.tensor([generation_start_pos for _ in range(batch_size)])
#     current_pos_tensor = ttnn.from_torch(
#         current_pos,
#         device=mesh_device,
#         dtype=ttnn.int32,
#         mesh_mapper=ttnn.ShardTensor2dMesh(
#             mesh_device,
#             dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
#             mesh_shape=model_args.cluster_shape,
#         ),
#     )
#     for i in range(generation_length):
#         logger.info(f"[Decoder] Generating token {i}")

#         # input = torch.randn(1, 32, 4096)
#         pt_decode_input = (
#             torch.rand(
#                 batch_size, seqlen, model_args.dim,
#                 # dtype=get_ref_model_dype(reference_model, model_args.model_name)
#                 dtype= torch.bfloat16
#             )
#             * 2
#         ) - 1
#         tt_decode_input = pt_decode_input.clone()

#         decode_input = model_args.prepare_residual_tensor_decode(
#             tt_decode_input,
#             # ttnn.DRAM_MEMORY_CONFIG,
#             model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
#         )

#         # Get cos/sin matrices for the current position of each user
#         rot_mats = rope_setup.get_rot_mats(current_pos)

#         # Run TT model
#         tt_out = tt_model(
#             decode_input,
#             current_pos_tensor,
#             rot_mats=rot_mats,
#             mode="decode",
#             page_table=page_table_tt,
#         )
#         tt_out = ttnn.to_torch(
#             tt_out,
#             mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
#         )

#         tt_output_torch = tt_out[:, 0:1, : model_args.max_batch_size, : model_args.dim].view(-1, 1, model_args.dim).squeeze(0).squeeze(0)
#         # print(tt_output_torch)
#         # In this test all users have the same position
#         freqs_cis_i = freqs_cis[current_pos[0], :].unsqueeze(0)

#         # Reference model
#         ref_output = reference_model(pt_decode_input, current_pos[0], freqs_cis_i, mask=None)

#         print(f"ttnn output: {tt_output_torch}")
#         print(f"Reference output: {ref_output}")

#         passing, pcc_message = comp_pcc(ref_output, tt_output_torch)

#         logger.info(comp_allclose(ref_output, tt_output_torch))
#         logger.info(f"PCC: {pcc_message}")

#         if passing:
#             logger.info("Decoder Block Passed!")
#         else:
#             logger.warning("Decoder Block Failed!")
#             all_tests_pass = False

#         # Increment position
#         current_pos = torch.tensor([generation_start_pos + i for _ in range(batch_size)])
#         current_pos_tensor = ttnn.from_torch(
#             current_pos,
#             device=mesh_device,
#             dtype=ttnn.int32,
#             mesh_mapper=ttnn.ShardTensor2dMesh(
#                 mesh_device,
#                 dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
#                 mesh_shape=model_args.cluster_shape,
#             ),
#         )

#     if all_tests_pass:
#         logger.info(f"All {generation_length} decode iterations Passed!")
#     else:
#         logger.warning("One or more iterations of decode Failed!")
#         assert all_tests_pass, f"PCC value is lower than {0.99} for some of the outputs. Check Warnings!"
