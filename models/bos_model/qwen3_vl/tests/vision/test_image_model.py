"""Test for Qwen 3 VL Vision Transformer Pretrained Model Inference"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.bos_model.qwen3_vl.tt.model_config import ModelArgs
from models.bos_model.qwen3_vl.tt.vision.qwen_vision_model import TtQwen3_VisionTransformerPretrainedModel
from models.common.utility_functions import comp_allclose, comp_pcc


# @skip_for_grayskull("Requires wormhole_b0 to run")
# @pytest.mark.skip(reason="Requires >4GB DRAM storage.")
@pytest.mark.parametrize(
    "batch, num_chunks",
    ((1, 4),),
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_vision_inference(batch, num_chunks, mesh_device, reset_seeds):
    dtype = ttnn.bfloat16
    pcc_required = 0.92
    all_tests_pass = True

    model_args = ModelArgs(mesh_device)
    state_dict = model_args.load_state_dict()

    first_layer_prefix = "visual."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    reference_model = model_args.reference_vision_model()
    reference_model.load_state_dict(partial_state_dict)
    reference_model.eval()

    n_layers = model_args.vision_n_layers

    tt_model = TtQwen3_VisionTransformerPretrainedModel(
        mesh_device,
        state_dict=state_dict,
        state_dict_prefix=first_layer_prefix,
        weight_cache_path=model_args.weight_cache_path(dtype),
        model_args=model_args,
        dtype=dtype,
        layers=n_layers,
    )

    # 3 (in_channels) * 2 (temporal_patch_size) * 16 (patch_size) * 16 (patch_size) = 1536
    patch_dim = (
        model_args.vision_in_channels
        * model_args.temporal_patch_size
        * model_args.vision_patch_size
        * model_args.vision_patch_size
    )

    # grid_thw = [t, h, w]. num_patches = t * h * w
    # Using 16x16 grid = 256 patches.
    grid_thw = torch.tensor([[1, 16, 16]])
    num_patches = grid_thw[0, 0].item() * grid_thw[0, 1].item() * grid_thw[0, 2].item()

    # Input to reference model is [NumPatches, PatchDim]
    pt_input = torch.rand(num_patches, patch_dim, dtype=torch.bfloat16)

    reference_output, ref_deepstack_outputs = reference_model(
        pt_input,
        grid_thw,
    )

    # tt_model.patch_embed expects [1, 1, NumPatches, PatchDim]
    tt_input_tensor = pt_input.reshape(1, 1, num_patches, patch_dim)

    tt_attention_input = ttnn.from_torch(
        tt_input_tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_out, tt_deepstack_outputs = tt_model(tt_attention_input, grid_thw)

    # tt_out is the final merged output, e.g., shape [64, 2560]
    tt_output_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    # Remove potential padding
    tt_output_torch = tt_output_torch[: tt_out.shape[0], :]

    logger.info("--- Comparing Main Vision Output ---")
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)
    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"Main Output PCC: {pcc_message}")
    if not passing:
        all_tests_pass = False

    logger.info(f"Found {len(ref_deepstack_outputs)} reference deepstack outputs.")
    logger.info(f"Found {len(tt_deepstack_outputs)} ttnn deepstack outputs.")

    assert len(ref_deepstack_outputs) == len(tt_deepstack_outputs), "Mismatch in number of deepstack outputs"

    for i, (ref_ds_out, tt_ds_out) in enumerate(zip(ref_deepstack_outputs, tt_deepstack_outputs)):
        logger.info(f"--- Comparing Deepstack Output {i} ---")

        tt_ds_out_torch = ttnn.to_torch(tt_ds_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        tt_ds_out_torch = tt_ds_out_torch[: tt_ds_out.shape[0], :]

        passing_ds, pcc_message_ds = comp_pcc(ref_ds_out, tt_ds_out_torch, pcc_required)
        logger.info(f"Deepstack Output {i} PCC: {pcc_message_ds}")

        if not passing_ds:
            all_tests_pass = False

    assert all_tests_pass, f"PCC value is lower than {pcc_required} for one or more outputs. Check Warnings!"
