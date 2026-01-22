# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
PyTest for the TT-NN implementation of SpatialCrossAttention + MSDeformableAttention3D.

• Compares TT results against the reference PyTorch implementation.
• Uses deterministic random inputs generated once per test run.
• Verifies shape equality and numerical closeness (PCC > 0.98).
"""

import time
import torch
import numpy as np
import pytest
import ttnn
from loguru import logger

from bos_metal import device_box, engine, op
from mmcv.cnn.bricks.transformer import build_attention

from test.SSR.reference.spatial_cross_attention import SpatialCrossAttention
from test.SSR.tt.spatial_cross_attention import SpatialCrossAttention


from test.utils import pt2tt, compare_tensors

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
    device_box.open(enable_program_cache=False)  # default device_id = 0
    dev = device_box.get()
    yield dev
    device_box.close()


@pytest.fixture
def random_inputs(device):
    """Generate deterministic PyTorch inputs and their TT equivalents."""

    # ------------------- PyTorch tensors ---------------------------------- #
    ref_query = torch.randn(BATCH_SIZE, SRC_SEQ_LEN, NUM_CHANNELS)
    ref_value = torch.randn(NUM_CAMS, TGT_SEQ_LEN, BATCH_SIZE, NUM_CHANNELS)
    ref_reference_points = torch.randn(BATCH_SIZE, 4, SRC_SEQ_LEN, 3)
    ref_reference_points_cam = torch.randn(NUM_CAMS, BATCH_SIZE, SRC_SEQ_LEN, 4, 2)
    ref_bev_mask = torch.randint(0, 2, (NUM_CAMS, BATCH_SIZE, SRC_SEQ_LEN, 4))

    level_start_index = torch.tensor([0])
    spatial_shapes = torch.tensor([[12, 20]])

    ref_input = dict(
        query=ref_query,
        key=ref_value,
        value=ref_value,
        reference_points=ref_reference_points,
        spatial_shapes=spatial_shapes,
        reference_points_cam=ref_reference_points_cam,
        bev_mask=ref_bev_mask,
        level_start_index=level_start_index,
    )

    # ------------------- TT-NN tensors ------------------------------------ #
    query = pt2tt(
        ref_query,
        # memory_config=ttnn.L1_MEMORY_CONFIG,
        # memory_config=ttnn.DRAM_MEMORY_CONFIG,
        memory_config=op.ShardedMemConfig((512, 256), (5, 4), "height", as_shard_shape=True),
        device=device,
    )
    value = pt2tt(ref_value.permute(2, 0, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG, device=device)  # B, C, T, D
    reference_points = pt2tt(ref_reference_points, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=device)
    reference_points_cam = pt2tt(ref_reference_points_cam, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=device)
    bev_mask = pt2tt(ref_bev_mask, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=device)

    tt_input = dict(
        query=query,
        key=value,
        value=value,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        reference_points_cam=reference_points_cam,
        bev_mask=bev_mask,
        level_start_index=level_start_index,
    )

    return ref_input, tt_input


# --------------------------------------------------------------------------- #
# Test                                                                        #
# --------------------------------------------------------------------------- #
def test_ms_deform_attn_3d(device, random_inputs):
    """End-to-end comparison between PyTorch and TT-NN SpatialCrossAttention."""

    ref_input, tt_input = random_inputs

    # ------------------- Build reference model ---------------------------- #
    cfg = dict(
        type="SpatialCrossAttention",
        pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        deformable_attention=dict(
            type="MSDeformableAttention3D",
            embed_dims=NUM_CHANNELS,
            num_points=8,
            num_levels=1,
        ),
        embed_dims=NUM_CHANNELS,
    )
    ref_model = build_attention(cfg)
    ref_model.eval()

    # ------------------- Build TT-NN model -------------------------------- #
    cfg_ttnn = cfg.copy()
    cfg_ttnn["type"] = "SpatialCrossAttention_tt"
    cfg_ttnn["deformable_attention"]["type"] = "MSDeformableAttention3D_tt"
    model = build_attention(cfg_ttnn)

    # Load processed state dict
    state_dict = engine.ModelProcessor(ref_model).process_state_dict(**ref_input)
    model.load_state_dict(state_dict, strict=False)

    # ------------------- Forward passes ----------------------------------- #
    tt_out = model(**tt_input)
    ref_out = ref_model(**ref_input)

    # ------------------- Assertions --------------------------------------- #
    assert ref_out.shape == tt_out.shape, "Output shape mismatch"
    pcc, _ = compare_tensors(ref_out, tt_out)
    assert pcc > 0.98, f"PCC below threshold: {pcc:.4f}"


    elapsed_times = []
    num_iters = 10
    for i in range(num_iters):
        tt_start = time.time()
        tt_out = model(**tt_input)

        # Ensure device synchronization before measuring time
        ttnn.synchronize_device(tt_out.device())

        tt_elapsed = time.time() - tt_start
        elapsed_times.append(tt_elapsed)

        logger.info(f"[Run {i+1}/{num_iters}] TT-NN forward pass {tt_elapsed:.3f}s " f"({1.0/tt_elapsed:.2f} FPS)")

    # Compute statistics
    avg_elapsed = float(np.mean(elapsed_times))
    fps = 1.0 / avg_elapsed if avg_elapsed > 0 else float("inf")

    logger.info(f"TT-NN forward pass avg: {avg_elapsed:.3f}s | {fps:.2f} FPS")
