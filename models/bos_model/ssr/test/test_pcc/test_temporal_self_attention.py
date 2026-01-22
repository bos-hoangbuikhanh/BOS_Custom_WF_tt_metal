# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
PyTest for the TT-NN implementation of TemporalSelfAttention.

• Compares TT results against the reference PyTorch implementation.
• Uses deterministic random inputs generated once per test run.
• Verifies shape equality and numerical closeness (PCC > 0.98).
"""

import time
from test.SSR.reference.temporal_self_attention import TemporalSelfAttention
from test.SSR.tt.temporal_self_attention import TemporalSelfAttention
from test.utils import compare_tensors, pt2tt
from tt.projects.configs.ops_config import memory_config, program_config

import pytest
import torch
from bos_metal import device_box, engine, op
from loguru import logger
from mmcv.cnn.bricks.transformer import build_attention

import ttnn

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #
torch.manual_seed(0)

BATCH_SIZE = 1
NUM_CAMS = 6
SRC_SEQ_LEN = 10_000  # dynamic length governed by bev_mask
TGT_SEQ_LEN = 240
NUM_CHANNELS = 256


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def device():
    """Open a Tenstorrent device for the whole test session, close at teardown."""
    device_box.open()  # default device_id = 0
    dev = device_box.get()
    yield dev
    device_box.close()


def random_inputs(device):
    """Generate deterministic PyTorch inputs and their TT equivalents."""

    # ------------------- PyTorch tensors ---------------------------------- #
    ref_query = torch.randn(BATCH_SIZE, SRC_SEQ_LEN, NUM_CHANNELS)
    ref_query_pos = torch.randn(BATCH_SIZE, SRC_SEQ_LEN, NUM_CHANNELS)
    ref_reference_points = torch.randn(2, SRC_SEQ_LEN, BATCH_SIZE, 2)

    level_start_index = torch.tensor([0])
    ref_spatial_shapes = torch.tensor([[100, 100]])

    ref_input = dict(
        query=ref_query,
        query_pos=ref_query_pos,
        reference_points=ref_reference_points,
        spatial_shapes=ref_spatial_shapes,
        level_start_index=level_start_index,
    )

    # ------------------- TT-NN tensors ------------------------------------ #
    query = pt2tt(
        ref_query,
        memory_config=op.ShardedMemConfig((512, 256), (5, 4), "height", as_shard_shape=True),
        device=device,
    )

    query_pos = pt2tt(ref_query_pos, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=device)

    reference_points = ref_reference_points.permute(1, 0, 2, 3)
    reference_points = reference_points.reshape(10_000, 1, 2, 2)
    reference_points = reference_points.repeat(1, 8, 1, 4)
    reference_points = reference_points.reshape(1, 10_000, 128)
    reference_points = pt2tt(reference_points, layout=ttnn.TILE_LAYOUT, device=device_box.get())
    spatial_shapes = pt2tt(
        ref_spatial_shapes, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    tt_input = dict(
        query=query,
        query_pos=query_pos,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )

    return ref_input, tt_input


# --------------------------------------------------------------------------- #
# Test                                                                        #
# --------------------------------------------------------------------------- #
def test_ms_deform_attn_3d(device):
    """End-to-end comparison between PyTorch and TT-NN SpatialCrossAttention."""

    ref_input, tt_input = random_inputs(device)

    # ------------------- Build reference model ---------------------------- #
    cfg = dict(type="TemporalSelfAttention", embed_dims=NUM_CHANNELS, num_levels=1)
    ref_model = build_attention(cfg)
    ref_model.eval()

    # ------------------- Build TT-NN model -------------------------------- #
    cfg_ttnn = cfg.copy()
    cfg_ttnn["type"] = "TemporalSelfAttention_tt"
    model = build_attention(cfg_ttnn)

    # Load processed state dict
    state_dict = engine.ModelProcessor(ref_model).process_state_dict(**ref_input)
    model.load_state_dict(state_dict, strict=False)

    # ------------------- Forward passes ----------------------------------- #
    tt_out = model( # Caching
        **tt_input,
        memory_config=memory_config["SSRHead"]["transformer"]["encoder"]["self_attn"],
        program_config=program_config["SSRHead"]["transformer"]["encoder"]["self_attn"],
    )
    ref_out = ref_model(**ref_input)

    # ------------------- Assertions --------------------------------------- #
    assert ref_out.shape == tt_out.shape, "Output shape mismatch"
    pcc, _ = compare_tensors(ref_out, tt_out)
    assert pcc > 0.98, f"PCC below threshold: {pcc:.4f}"
    breakpoint()
    import numpy as np

    num_runs = 10
    times = []

    for i in range(num_runs):
        tt_start = time.time()
        tt_out = model(
            **tt_input,
            memory_config=memory_config["SSRHead"]["transformer"]["encoder"]["self_attn"],
            program_config=program_config["SSRHead"]["transformer"]["encoder"]["self_attn"],
        )
        ttnn.synchronize_device(device)
        tt_elapsed = time.time() - tt_start
        times.append(tt_elapsed)
        logger.info(f"[Run {i+1}/{num_runs}] TT-NN forward pass: {tt_elapsed:.6f} s")

    avg_time = np.mean(times)
    std_time = np.std(times)
    logger.info(
        f"TT-NN forward pass over {num_runs} runs: "
        f"avg = {avg_time:.6f} s, std = {std_time:.6f} s, "
        f"min = {np.min(times):.6f} s, max = {np.max(times):.6f} s"
    )
