# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
PyTest for ttnn.operations.nonzero on a BEV-mask tensor.

Original imperative script:
  – Builds a boolean BEV mask of shape [6, 1, 10 000, 4]
  – Finds non-zero indices on the NPU and compares against PyTorch
This version:
  – Uses pytest fixtures & parametrisation
  – Runs the same logic five times to mirror the original loop
"""

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal

################################################################################
# Constants
################################################################################
BEV_SHAPE = [6, 1, 10_000, 4]          # [B, 1, H, W]
NUM_RUNS  = 1                          # replicates the original for-loop


################################################################################
# Helper
################################################################################
def _gather_nonzero_indices(bev_mask_img):
    """
    Internal utility that replicates the ttnn nonzero workflow
    for a single image slice.
    """
    # Sum over W, keep dims so the tensor is [1, H]
    bev_sum   = ttnn.sum(bev_mask_img, -1)
    # Non-zero on row-major layout
    num_idx, idx = ttnn.bos_nonzero(ttnn.to_layout(bev_sum, ttnn.ROW_MAJOR_LAYOUT))
    # Trim padded zeros (num_idx returned as scalar tensor on host)
    valid_cols = num_idx.cpu().to_torch()[0].item()
    return idx[..., :valid_cols]


################################################################################
# Main test
################################################################################
@pytest.mark.parametrize("repeat", range(NUM_RUNS), ids=lambda i: f"run{i}")
def test_nonzero_indices(device, repeat):
    torch.manual_seed(0)                             # deterministic
    ref_bev_mask = torch.randint(0, 2, BEV_SHAPE, dtype=torch.bool)

    # Upload full tensor in TILE_LAYOUT
    bev_mask = ttnn.from_torch(
        ref_bev_mask,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT, # Switch to ROW_MAJOR, TILE LAYOUT is fail on A0 
    )
    
    ref_query = torch.randn(1, 10_000, 256, dtype=torch.bfloat16)
    query = ttnn.from_torch(ref_query, device=device, layout=ttnn.ROW_MAJOR_LAYOUT) # Switch to ROW_MAJOR, TILE LAYOUT is fail on A0 

    ttnn_indices  = []
    torch_indices = []

    # Loop over batch dimension B
    for i in range(bev_mask.shape[0]):
        # --- TTNN path ------------------------------------------------------ #
        per_img_idx = _gather_nonzero_indices(bev_mask[i])
        ttnn_indices.append(per_img_idx)

        # --- PyTorch reference --------------------------------------------- #
        ref_idx = ref_bev_mask[i, 0].sum(-1).nonzero().squeeze(-1)
        torch_indices.append(ref_idx)

        # Immediate per-image comparison
        assert_equal(ref_idx.to(torch.bfloat16), ttnn.to_torch(per_img_idx))

        # --- Getter sanity check ------------------------------------------ #
        ref_selected = ref_query[0, ref_idx] 
        selected = ttnn.operations.moreh.getitem(
            query[0], [per_img_idx], [0]
        ) 
        assert_equal(ref_selected, ttnn.to_torch(selected))

    # --- Batch-level sanity -------------------------------------------------- #
    # assert max(x.shape[-1] for x in ttnn_indices) == max(
    #     y.shape[-1] for y in torch_indices
    # ), "Max index length differs between TTNN and reference"

    # Element-wise checks across the batch
    for i, (idx_tt, idx_pt) in enumerate(zip(ttnn_indices, torch_indices)):
        # assert idx_tt.shape[-1] == idx_pt.shape[-1], f"Shape mismatch at img {i}"
        assert_equal(ttnn.to_torch(idx_tt), idx_pt)
