"""Test for Qwen 3 VL Vision Rotary Embedding"""

import os

import pytest
import torch
import torch.nn as nn
from loguru import logger

import ttnn
from models.bos_model.qwen3_vl.tt.model_config import ModelArgs
from models.bos_model.qwen3_vl.tt.vision.qwen_rope import TTQwen3_VisionRotaryEmbedding
from models.common.utility_functions import comp_allclose, comp_pcc


class PtQwen3VLVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


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
    (16, 256, 1024),
)
def test_rope_inference(seq_len, reset_seeds, device):
    dtype = ttnn.bfloat16
    pcc_required = 0.9999

    model_args = ModelArgs(device)

    # The RoPE dim is half the head dim
    vision_dim = model_args.vision_dim
    n_heads = model_args.vision_attn_n_heads
    head_dim = vision_dim // n_heads
    rope_dim = head_dim // 2
    theta = 10000.0

    reference_model = PtQwen3VLVisionRotaryEmbedding(dim=rope_dim, theta=theta)
    reference_model.eval()

    tt_model = TTQwen3_VisionRotaryEmbedding(
        device=device,
        dim=rope_dim,
        theta=theta,
    )
    reference_output = reference_model(seq_len)  # Shape [seq_len, 32]

    tt_out = tt_model(seq_len)  # Shape [seq_len, 32]

    tt_output_torch = ttnn.to_torch(tt_out)

    tt_output_torch = tt_output_torch.reshape(reference_output.shape)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    if passing:
        logger.info("VisionRoPE Passed!")
    else:
        logger.warning("VisionRoPE Failed!")

    assert passing, f"VisionRoPE output does not meet PCC requirement {pcc_required}."
