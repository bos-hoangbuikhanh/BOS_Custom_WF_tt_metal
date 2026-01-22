"""Test for Qwen 3 VL DeepStack Vision Patch Merger"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.bos_model.qwen3_vl.tt.model_config import ModelArgs
from models.bos_model.qwen3_vl.tt.vision.qwen_patch_merger import TTQwen3_VLPatchMerger
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
    (128,),  # This is the number of patches, e.g., 16x8. Must be multiple of 4.
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
def test_deepstack_merger_inference(seq_len, batch_size, reset_seeds, device):
    dtype = ttnn.bfloat8_b
    pcc_required = 0.99

    tt_model_args = ModelArgs(
        device,
        max_batch_size=batch_size,
        max_seq_len=seq_len,
    )

    tt_model_args.n_layers = 1
    state_dict = tt_model_args.load_state_dict()

    reference_model = tt_model_args.reference_vision_deepstack_merger()

    first_layer_prefix = "visual.deepstack_merger_list.0."

    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    reference_model.load_state_dict(partial_state_dict)
    reference_model.eval()

    tt_model = TTQwen3_VLPatchMerger(
        mesh_device=device,
        args=tt_model_args,
        state_dict=state_dict,
        weight_cache_path=tt_model_args.weight_cache_path(dtype),
        dtype=dtype,
        state_dict_prefix=first_layer_prefix,
        use_postshuffle_norm=True,
    )

    input_dim = tt_model_args.vision_dim
    assert seq_len % 4 == 0, "Sequence length (num_patches) must be a multiple of 4 for merging"

    pt_input_tensor = torch.rand(seq_len, input_dim, dtype=torch.bfloat16)

    reference_output = reference_model(pt_input_tensor)

    tt_input_tensor_4d = pt_input_tensor.reshape(1, 1, seq_len, input_dim)

    tt_input = ttnn.from_torch(
        tt_input_tensor_4d,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = tt_model(tt_input)

    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
    tt_output_torch = tt_output_torch[: tt_output.shape[0], :]

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    print(f"Reference output: {reference_output}, shape: {reference_output.shape}")
    print(f"TTNN output: {tt_output_torch}, shape: {tt_output_torch.shape}")

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    if passing:
        logger.info("DeepStack Merger Passed!")
    else:
        logger.warning("DeepStack Merger Failed!")

    assert passing, f"DeepStack Merger output does not meet PCC requirement {pcc_required}."
