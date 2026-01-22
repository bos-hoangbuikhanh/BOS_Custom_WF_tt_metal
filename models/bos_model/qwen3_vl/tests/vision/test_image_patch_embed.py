"""Test for Qwen 3 VL Vision Patch Embed"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.bos_model.qwen3_vl.tt.model_config import ModelArgs
from models.bos_model.qwen3_vl.tt.vision.qwen_image_patch_embed import TTQwen3_VisionPatchEmbed
from models.common.utility_functions import comp_allclose, comp_pcc


@torch.no_grad()
# @skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("device"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len",
    (128,),  # This will be the number of patches
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
def test_embed_inference(seq_len, batch_size, reset_seeds, device):
    dtype = ttnn.bfloat8_b
    pcc_required = 0.99
    mode = "decode" if seq_len <= 32 else "prefill"

    tt_model_args = ModelArgs(
        device,
        max_batch_size=batch_size,
        max_seq_len=128,
    )

    tt_model_args.n_layers = 1
    state_dict = tt_model_args.load_state_dict()

    reference_model = tt_model_args.reference_vision_qwen_patch_embed()
    first_layer_prefix = "visual.patch_embed."

    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    reference_model.load_state_dict(partial_state_dict)
    reference_model.eval()

    tt_model = TTQwen3_VisionPatchEmbed(
        device=device,
        args=tt_model_args,
        patch_size=16,
        temporal_patch_size=2,
        in_channels=3,
        embed_dim=1024,
        state_dict=state_dict,
        weight_key=first_layer_prefix,
        layer_num=None,
        state_dict_prefix="",
        weight_cache_path=None,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        weight_dtype=dtype,
        mode=mode,
    )

    patch_dim = 3 * 2 * 16 * 16

    # Shape is [Batch, 1, NumPatches, PatchDim]
    # The reference model's view() will correctly interpret this
    input_tensor = torch.rand(batch_size, 1, seq_len, patch_dim, dtype=torch.bfloat16)

    # Run reference
    reference_output = reference_model(input_tensor)
    # Ref output shape is [NumPatches, EmbedDim] = [128, 1024]

    tt_input = ttnn.from_torch(
        input_tensor,
        device=device,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = tt_model(tt_input)
    # TT output shape is [NumPatches, EmbedDim] = [128, 1024]

    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=-1))
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    print(f"Reference output:{reference_output}")
    print(f"TTNN output: {tt_output_torch}")

    if passing:
        logger.info("Patch embed Passed!")
    else:
        logger.warning("Patch embed Failed!")

    assert passing, f"Patch embed output does not meet PCC requirement {pcc_required}."
