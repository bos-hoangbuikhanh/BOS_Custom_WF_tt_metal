"""Test for Qwen 3 VL Vision Patch Merger"""  # <--- CHANGED

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
    (128,),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
def test_patch_merger_inference(seq_len, batch_size, reset_seeds, device):
    dtype = ttnn.bfloat8_b
    pcc_required = 0.99

    tt_model_args = ModelArgs(
        device,
        max_batch_size=batch_size,
        max_seq_len=128,
    )

    tt_model_args.n_layers = 1
    state_dict = tt_model_args.load_state_dict()

    reference_model = tt_model_args.reference_vision_qwen_merger()
    first_layer_prefix = "visual.merger."

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
    )

    input_dim = tt_model_args.vision_dim

    input_seq_len = 4
    input_tensor = torch.rand(batch_size, input_seq_len, input_dim, dtype=torch.bfloat16)

    reference_output = reference_model(input_tensor)

    tt_input_tensor = input_tensor.reshape(1, 1, input_seq_len, input_dim)

    tt_input = ttnn.from_torch(
        tt_input_tensor,
        device=device,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = tt_model(tt_input)

    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))

    if tt_output_torch.dim() > 2:
        tt_output_torch = tt_output_torch.squeeze(0).squeeze(0)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    print(f"Reference output:{reference_output}")
    print(f"TTNN output: {tt_output_torch}")

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    if passing:
        logger.info("Merger Passed!")
    else:
        logger.warning("Merger Failed!")

    assert passing, f"Merger output does not meet PCC requirement {pcc_required}."
