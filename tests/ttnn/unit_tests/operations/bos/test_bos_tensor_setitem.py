import torch
import pytest
import ttnn
from loguru import logger
from models.common.utility_functions import comp_pcc

SHAPES = [
    (8,),  # 1-D
    (6, 12),  # 2-D
    (2, 5, 10),  # 3-D
    (1, 3, 4, 50),  # 4-D
    (1, 3, 20, 5),  # 4-D
    (1, 3, 21, 5),  # 4-D
    (1, 21, 3, 5),  # 4-D
]
POS = ["start", "middle", "end"]


def _idx_pos(size):
    return torch.randint(0, size, ()).item()


def _slice_pos(size, pos):
    if pos == "start":
        return slice(0, max(1, size // 3))
    if pos == "end":
        k = max(1, size - size // 3)
        return slice(k, size)
    # middle
    k1 = size // 3
    return slice(k1, min(size, k1 + max(1, size // 3)))


def _build_src(shape):
    return torch.arange(torch.tensor(shape).prod(), dtype=torch.bfloat16).reshape(shape)


def _to_ttnn(t, dev):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=dev)


# =======================================================================
# @pytest.mark.skip("Not supported yet")
@pytest.mark.parametrize("shape", [s for s in SHAPES])
def test_index_only(shape, device):
    """
    Rank 1-5:
      A[i0], A[i0,i1], A[i0,i1,i2], …, A[i0,i1,i2,i3,i4]
    """
    # torch.manual_seed(0)
    rank = len(shape)
    idxs = [_idx_pos(sz) for sz in shape]

    # build reference
    x_ref = torch.zeros(shape, dtype=torch.bfloat16)
    src = _build_src(shape)
    # Get index by rank
    if rank == 1:
        expected = src[idxs[0]]
    elif rank == 2:
        expected = src[idxs[0], idxs[1]]
    elif rank == 3:
        expected = src[idxs[0], idxs[1], idxs[2]]
    elif rank == 4:
        expected = src[idxs[0], idxs[1], idxs[2], idxs[3]]
    else:  # rank == 5
        expected = src[idxs[0], idxs[1], idxs[2], idxs[3], idxs[4]]
    x_ref[...] = 0  # reset
    # set item
    if rank == 1:
        x_ref[idxs[0]] = expected
    elif rank == 2:
        x_ref[idxs[0], idxs[1]] = expected
    elif rank == 3:
        x_ref[idxs[0], idxs[1], idxs[2]] = expected
    elif rank == 4:
        x_ref[idxs[0], idxs[1], idxs[2], idxs[3]] = expected
    else:
        x_ref[idxs[0], idxs[1], idxs[2], idxs[3], idxs[4]] = expected

    # TT-NN
    x_tt = _to_ttnn(torch.zeros_like(x_ref), device)
    exp_tt = _to_ttnn(expected.clone(), device)

    if rank == 1:
        x_tt[idxs[0]] = exp_tt
    elif rank == 2:
        x_tt[idxs[0], idxs[1]] = exp_tt
    elif rank == 3:
        x_tt[idxs[0], idxs[1], idxs[2]] = exp_tt
    elif rank == 4:
        x_tt[idxs[0], idxs[1], idxs[2], idxs[3]] = exp_tt
    else:
        x_tt[idxs[0], idxs[1], idxs[2], idxs[3], idxs[4]] = exp_tt

    x_tt_rs = ttnn.to_torch(x_tt)
    ok, diff = comp_pcc(x_ref, x_tt_rs)
    logger.info(f"[index] shape={shape}, idxs={idxs}\n{diff}")
    # assert ok, f"PCC={diff:.4f}, shape={shape}, x_ref={x_ref}, x_tt={x_tt_rs}"
    assert torch.allclose(x_ref, x_tt_rs), f"Not allclose: shape={shape}, x_ref={x_ref}, x_tt={x_tt_rs}"


# =======================================================================
@pytest.mark.parametrize("shape, pos", [(s, p) for s in SHAPES for p in POS])
def test_slice_only(shape, pos, device):
    """
    - rank=1: A[:k] / A[k1:k2] / A[k:]
    - rank>=2: A[:, ..., slice, :]
    """
    torch.manual_seed(0)
    rank = len(shape)
    seq_dim = 0 if rank == 1 else -2
    sl = _slice_pos(shape[seq_dim], pos)

    # build reference
    x_ref = torch.zeros(shape, dtype=torch.bfloat16)
    src = _build_src(shape)
    # Set item
    if rank == 1:
        expected = src[sl]
        x_ref[sl] = expected
    elif rank == 2:
        expected = src[sl, :]
        x_ref[sl, :] = expected
    elif rank == 3:
        expected = src[:, sl, :]
        x_ref[:, sl, :] = expected
    elif rank == 4:
        expected = src[:, :, sl, :]
        x_ref[:, :, sl, :] = expected
    else:  # rank == 5
        expected = src[:, :, :, sl, :]
        x_ref[:, :, :, sl, :] = expected

    # TT-NN
    x_tt = _to_ttnn(torch.zeros_like(x_ref), device)
    exp_tt = _to_ttnn(expected.clone(), device)
    if rank == 1:
        x_tt[sl] = exp_tt
    elif rank == 2:
        x_tt[sl, :] = exp_tt
    elif rank == 3:
        x_tt[:, sl, :] = exp_tt
    elif rank == 4:
        x_tt[:, :, sl, :] = exp_tt
    else:
        x_tt[:, :, :, sl, :] = exp_tt

    x_tt_rs = ttnn.to_torch(x_tt)
    ok, diff = comp_pcc(x_ref, x_tt_rs)
    logger.info(f"[slice] shape={shape}, pos={pos}, slice={sl}\n{diff}")
    # assert ok, f"PCC={diff:.4f}, shape={shape}, pos={pos}, x_ref={x_ref}, x_tt={x_tt_rs}"
    print(f"x_ref={x_ref}")
    print(f"x_tt_rs={x_tt_rs}")
    # assert torch.allclose(x_ref, x_tt_rs), f"Not allclose: shape={shape}, pos={pos}, x_ref={x_ref}, x_tt={x_tt_rs}"


# =======================================================================
@pytest.mark.parametrize("shape, pos", [(s, p) for s in SHAPES for p in POS])
def test_with_ellipsis_with_slice(shape, pos, device):
    """
    A[..., i] (rank=1)
    A[..., i_seq, i_emb] (rank>=2)
    """
    torch.manual_seed(0)
    rank = len(shape)
    # pick positions
    seq_dim = 0 if rank == 1 else -2
    i_seq = _slice_pos(shape[seq_dim], pos)
    i_embed = _slice_pos(shape[-1], pos)

    # build reference
    x_ref = torch.zeros(shape, dtype=torch.bfloat16)
    src = _build_src(shape)
    if rank == 1:
        expected = src[..., i_seq]
        x_ref[..., i_seq] = expected
    else:
        expected = src[..., i_seq, i_embed]
        x_ref[..., i_seq, i_embed] = expected

    # TT-NN
    x_tt = _to_ttnn(torch.zeros_like(x_ref), device)
    exp_tt = _to_ttnn(expected.clone(), device)
    if rank == 1:
        x_tt[..., i_seq] = exp_tt
    else:
        x_tt[..., i_seq, i_embed] = exp_tt

    x_tt_rs = ttnn.to_torch(x_tt)
    ok, pcc_val = comp_pcc(x_ref, x_tt_rs)
    # assert ok, f"PCC={pcc_val:.4f}, shape={shape}, pos={pos}, x_ref={x_ref}, x_tt={x_tt_rs}"
    assert torch.allclose(x_ref, x_tt_rs), f"Not allclose: shape={shape}, pos={pos}, x_ref={x_ref}, x_tt={x_tt_rs}"


# =======================================================================
# @pytest.mark.skip("Not supported yet")
@pytest.mark.parametrize("shape, pos", [(s, p) for s in SHAPES for p in POS])
def test_with_ellipsis(shape, pos, device):
    """
    A[..., i] (rank=1)
    A[..., i_seq, i_emb] (rank>=2)
    """
    torch.manual_seed(0)
    rank = len(shape)
    # pick positions
    seq_dim = 0 if rank == 1 else -2
    i_seq = _idx_pos(shape[seq_dim])
    i_embed = _slice_pos(shape[-1], pos)

    # build reference
    x_ref = torch.zeros(shape, dtype=torch.bfloat16)
    src = _build_src(shape)
    if rank == 1:
        expected = src[..., i_seq]
        x_ref[..., i_seq] = expected
    else:
        expected = src[..., i_seq, i_embed]
        x_ref[..., i_seq, i_embed] = expected

    # TT-NN
    x_tt = _to_ttnn(torch.zeros_like(x_ref), device)
    exp_tt = _to_ttnn(expected.clone(), device)
    if rank == 1:
        x_tt[..., i_seq] = exp_tt
    else:
        x_tt[..., i_seq, i_embed] = exp_tt

    x_tt_rs = ttnn.to_torch(x_tt)
    ok, pcc_val = comp_pcc(x_ref, x_tt_rs)
    # assert ok, f"PCC={pcc_val:.4f}, shape={shape}, pos={pos}, \nx_ref={x_ref}, \nx_tt={x_tt_rs}"
    assert torch.allclose(x_ref, x_tt_rs), f"Not allclose: shape={shape}, pos={pos}, x_ref={x_ref}, x_tt={x_tt_rs}"


# if __name__ == "__main__":
#     device = ttnn.open_device(device_id=0)
#     test_slice_only(SHAPES[0], POS[0], device)
#     test_slice_only(SHAPES[0], POS[1], device)
#     test_slice_only(SHAPES[0], POS[2], device)
