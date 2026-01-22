# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
PyTest coverage for `ttnn.operations.moreh.getitem`.

Tensor shape : [1, 10_000, 256]
Gather axis   : 1      (sequence length)
Index count   : 2 986
"""

import torch
import pytest
import ttnn
from loguru import logger

from bos_metal import device_box
from test.utils import compare_tensors  # project-specific helper

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SET_ITEM_SHAPE = [1, 6, 3680, 256]
SEQ_SHAPE  = [1, 10_000, 256]
INDEX_DIM  = 1
INDEX_SIZE = 3680 # 2_986
NUM_RUNS   = 1            # replicates the original for-loop
torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def device():
    """Open Tenstorrent device once per pytest session and close on teardown."""
    device_box.open()
    dev = device_box.get()
    yield dev
    device_box.close()


# --------------------------------------------------------------------------- #
# Helper – the core test logic
# --------------------------------------------------------------------------- #
def _run_getitem(device, dtype: torch.dtype):
    tt_dtype = ttnn.int32 if dtype is torch.int32 else ttnn.bfloat16

    # ---------- build input & indices ------------------------------------- #
    x       = torch.randint(0, 10, SEQ_SHAPE, dtype=dtype)
    idx_max = SEQ_SHAPE[INDEX_DIM] - 1
    idx     = torch.randint(-idx_max - 1, idx_max, (INDEX_SIZE,))

    dev_x   = ttnn.Tensor(x, tt_dtype).to(device)
    dev_idx = ttnn.Tensor(idx, ttnn.int32).to(device)
    dev_x = ttnn.to_layout(dev_x, ttnn.ROW_MAJOR_LAYOUT) # Switch to ROW_MAJOR, TILE LAYOUT is fail on A0 

    # ---------- reference -------------------------------------------------- #
    ref = x[:, idx]                              # (1, 2 986, 256)

    # ---------- NPU result ------------------------------------------------- #
    import tracy
    tracy.signpost("moreh")
    npu_out = (
        ttnn.operations.moreh.getitem(dev_x, [dev_idx], [INDEX_DIM])
        .cpu()
        .to_torch()
    )

    tracy.signpost("bos")
    npu_out = (
        ttnn.bos_getitem(dev_x, [dev_idx], [INDEX_DIM])
        .cpu()
        .to_torch()
    )

    # ---------- compare ---------------------------------------------------- #
    _, score = compare_tensors(ref.to(torch.bfloat16), npu_out.to(torch.bfloat16))
    assert score > 0.9999, f"PCC too low: {score:0.6f}"
    assert torch.allclose(ref, npu_out, atol=0, rtol=0), "Exact mismatch"

    logger.info(f"getitem OK — max abs diff: {(ref - npu_out).abs().max().item():.3e}")

def _run_get_setitem(device, dtype: torch.dtype):
    tt_dtype = ttnn.int32 if dtype is torch.int32 else ttnn.bfloat16

    # ---------- build input & indices ------------------------------------- #
    queries_rebatch = torch.zeros(SET_ITEM_SHAPE, dtype=torch.bfloat16)
    query = torch.randn(SEQ_SHAPE, dtype=torch.bfloat16)
    index_query_per_img = torch.randint(4950, 6400, (INDEX_SIZE,))

    queries_rebatch_tt = ttnn.Tensor(queries_rebatch, ttnn.bfloat16).to(device)
    queries_rebatch_tt = ttnn.to_layout(queries_rebatch_tt, ttnn.TILE_LAYOUT)
    query_tt = ttnn.Tensor(query, ttnn.bfloat16).to(device)
    query_tt = ttnn.to_layout(query_tt, ttnn.TILE_LAYOUT)
    index_query_per_img_tt = ttnn.Tensor(index_query_per_img, ttnn.int32).to(device)
    index_query_per_img_tt = ttnn.to_layout(index_query_per_img_tt, ttnn.ROW_MAJOR_LAYOUT)

    # ---------- reference -------------------------------------------------- #
    queries_rebatch[0, 0] = query[0][[index_query_per_img]]

    # ---------- NPU result ------------------------------------------------- #
    queries_rebatch_tt = ttnn.to_layout(queries_rebatch_tt, ttnn.ROW_MAJOR_LAYOUT)
    query_tt = ttnn.to_layout(query_tt, ttnn.ROW_MAJOR_LAYOUT)
    import tracy
    tracy.signpost("moreh")
    queries_rebatch_tt[0, 0] = ttnn.operations.moreh.getitem(query_tt[0], [index_query_per_img_tt], [0])
    tracy.signpost("bos")
    queries_rebatch_tt[0, 0] = ttnn.bos_getitem(query_tt[0], [index_query_per_img_tt], [0])
    queries_rebatch_tt = ttnn.to_torch(queries_rebatch_tt)

    # ---------- compare ---------------------------------------------------- #
    _, score = compare_tensors(queries_rebatch[0, 0].to(torch.bfloat16), queries_rebatch_tt[0, 0].to(torch.bfloat16))
    assert score > 0.9999, f"PCC too low: {score:0.6f}"
    _, score = compare_tensors(queries_rebatch.to(torch.bfloat16), queries_rebatch_tt.to(torch.bfloat16))
    assert score > 0.9999, f"PCC too low: {score:0.6f}"

# --------------------------------------------------------------------------- #
# Parametrised tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.int32], ids=["bfloat16", "int32"])
@pytest.mark.parametrize("repeat", range(NUM_RUNS), ids=lambda i: f"run{i}")
def test_getitem_long_sequence(device, dtype, repeat):
    """Run NUM_RUNS×2 (dtype) iterations to mimic original script behaviour."""
    _run_get_setitem(device, dtype)
