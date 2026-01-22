"""Test for Qwen 3 VL LayerNorm Layer Inference"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.bos_model.qwen3_vl.tt.model_config import ModelArgs
from models.bos_model.qwen3_vl.tt.vision.qwen_layernorm import TtVisionLayerNorm
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
def test_layernorm_inference(seq_len, batch_size, reset_seeds, device):
    dtype = ttnn.bfloat8_b
    pcc_required = 0.99

    tt_model_args = ModelArgs(
        device,
        max_batch_size=batch_size,
        max_seq_len=seq_len,
    )

    dim = tt_model_args.vision_dim

    state_dict = tt_model_args.load_state_dict()

    # Use the LayerNorm reference model
    reference_model = tt_model_args.reference_vision_layernorm()
    first_layer_prefix = "visual.blocks.0.norm1."  # Test norm1 of block 0

    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    reference_model.load_state_dict(partial_state_dict)
    reference_model.eval()

    tt_model = TtVisionLayerNorm(
        device=device,
        dim=dim,
        state_dict=state_dict,
        state_dict_prefix=first_layer_prefix,
        dtype=ttnn.bfloat16,
        eps=1e-6,
    )

    # Input shape is [B, S, D]
    input_tensor = torch.rand(batch_size, seq_len, dim, dtype=torch.bfloat16)

    reference_output = reference_model(input_tensor)

    # TT input must be 4D for TILE layout
    # Reshape to [1, B*S, D] and pad to [1, 1, B*S_padded, D_padded]
    tt_input_tensor = input_tensor.reshape(1, 1, batch_size * seq_len, dim)

    tt_input = ttnn.from_torch(
        tt_input_tensor,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = tt_model(tt_input)

    tt_output_torch = ttnn.to_torch(tt_output)

    tt_output_torch = tt_output_torch.reshape(batch_size, seq_len, dim)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    if passing:
        logger.info("LayerNorm Passed!")
    else:
        logger.warning("LayerNorm Failed!")

    assert passing, f"LayerNorm output does not meet PCC requirement {pcc_required}."
