"""
Test cases:
1. Test setitem kernels with explicit index tensors.

2. Test setitem with slices (torch-like api).
    - Same Value-input dtype: BF16-BF16, F32-F32, I32-I32, I64-I64.
    - Row Major, Interleaved.
    - Different Value-input dtype: F32-BF16, BF16-F32, I64-I32, I32-I64, F32-I64. #TODO
    - ...

    - Slice outer, full inner.
    - Slice outer, misaligned inner.
    - Input shape = Output shape
    - Empty slice.
    - Broadcast stage (same rank, lower-rank)
"""
import pytest
import torch
import ttnn

f16 = torch.float16
f32 = torch.float32
i32 = torch.int32
i64 = torch.int64
bf16 = torch.bfloat16

##################
#### Helper ######
##################


def make_input(shape=(4, 4), dtype=bf16, layout=ttnn.TILE_LAYOUT, device=None):
    tensor_pt = torch.zeros(shape, dtype=dtype)
    return ttnn.from_torch(tensor_pt, device=device, layout=layout), tensor_pt


def make_value(shape=(1, 4), dtype=bf16, layout=ttnn.TILE_LAYOUT, device=None):
    num_values = torch.tensor(shape).prod().item()
    tensor_pt = torch.arange(0, num_values, dtype=dtype).reshape(shape)
    return ttnn.from_torch(tensor_pt, device=device, layout=layout), tensor_pt


def make_index(value, dtype=i32, device=None):
    tensor_pt = torch.tensor(value, dtype=dtype)
    return ttnn.from_torch(tensor_pt, device=device), tensor_pt


def capture_error_type(error):
    """
    Capture the type of an error.
    """
    return type(error)


def capture_error_msg(error):
    """
    Capture the message of an error.
    """
    return str(error)


# context manager capture error
ERROR_MAPPING = {
    "UNSUPPORTED_LOWER_DIM_BROADCAST": "Lower-dim broadcasting is not implicitly supported",
    "UNSUPPORTED_NEGATIVE_STEP": "Negative step is not supported",
    "NOT_BROADCASTABLE": "not broadcastable to",
}


class raises_with_msg:
    def __init__(self, expected_exception, match_substring=None):
        self.expected_exception = expected_exception
        self.match_substring = match_substring
        self._pytest_raises_ctx = None

    def __enter__(self):
        self._pytest_raises_ctx = pytest.raises(self.expected_exception)
        return self._pytest_raises_ctx.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        result = self._pytest_raises_ctx.__exit__(exc_type, exc_val, exc_tb)

        if not result:
            return result  # propagate failure from pytest.raises

        if self.match_substring is not None:
            actual_msg = str(exc_val)
            # print(actual_msg)
            assert self.match_substring in actual_msg, (
                f"Expected error message to include '{self.match_substring}', " f"but got: '{actual_msg}'"
            )

        return True  # suppress exception


#############################
#### Setitem with slices ####
#############################


def setitem_slice(
    device, inp_shape, inp_dtype, val_shape, val_dtype, begins, ends, steps, idx_dtype, layout=ttnn.TILE_LAYOUT
):
    # prepare input & value
    att, apt = make_input(inp_shape, device=device, layout=layout, dtype=inp_dtype)
    vtt, vpt = make_value(val_shape, dtype=val_dtype, layout=layout, device=device)

    # --- Slice logic ---
    slices = []
    adjusted_begins = list(begins)
    adjusted_ends = list(ends)
    adjusted_steps = list(steps)

    # build slices
    for b, e, s in zip(adjusted_begins, adjusted_ends, adjusted_steps):
        slices.append(slice(b, e, s))
    slices = tuple(slices)

    apt[slices] = vpt
    # call via the new slice-API
    ott = ttnn.bos_setitem(
        att,
        vtt,
        begins=begins,
        ends=ends,
        steps=steps,
    )
    result = ttnn.to_torch(ott)

    print(result, apt)

    # 5) compare
    assert torch.allclose(result, apt), f"Setitem failed\nExpected:\n{apt}\nGot:\n{result}"


# Slice outer, full inner
@pytest.mark.parametrize("inp_dtype, val_dtype, idx_dtype", [(bf16, bf16, i32)])
@pytest.mark.parametrize(
    "inp_shape, val_shape, begins, ends, steps",
    [
        # === 2D tensors ===
        # Small volume, full slice (no broadcast)
        ((4, 4), (4, 4), [0, 0], [4, 4], [1, 1]),
        ((32, 53), (32, 53), [0, 0], [32, 53], [1, 1]),
        ((32, 64), (32, 64), [0, 0], [32, 64], [1, 1]),
        # Medium to large volume, full slice (no broadcast)
        ((256, 768), (256, 768), [0, 0], [256, 768], [1, 1]),
        ((1024, 1291), (1024, 1291), [0, 0], [1024, 1291], [1, 1]),
        # Partial slice (no broadcast), value aligned or matching input
        ((32, 53), (32, 32), [0, 0], [32, 32], [1, 1]),
        ((32, 64), (32, 32), [0, 0], [32, 32], [1, 1]),
        ((32, 64), (32, 32), [0, 32], [32, 64], [1, 1]),
        ((256, 768), (64, 768), [0, 0], [64, 768], [1, 1]),
        ((256, 768), (64, 768), [32, 0], [96, 768], [1, 1]),
        ((1024, 1291), (128, 1291), [0, 0], [128, 1291], [1, 1]),
        ((1024, 1291), (128, 1291), [64, 0], [192, 1291], [1, 1]),
        # Aligned partial slices (tile-aligned value and begins)
        ((1024, 1291), (128, 32), [0, 0], [128, 32], [1, 1]),
        ((1024, 1291), (128, 32), [64, 32], [192, 64], [1, 1]),
        ((64, 1291), (32, 64), [32, 224], [64, 288], [1, 1]),
        # === 3D tensors ===
        # Small volume, full slice (no broadcast)
        ((3, 4, 5), (3, 4, 5), [0, 0, 0], [3, 4, 5], [1, 1, 1]),
        ((3, 4, 5), (1, 4, 5), [0, 0, 0], [3, 4, 5], [1, 1, 1]),  # broadcast
        # Medium volume
        ((34, 32, 53), (34, 32, 53), [0, 0, 0], [34, 32, 53], [1, 1, 1]),
        # Large volume, partial slice
        ((4, 128, 1291), (2, 32, 64), [1, 32, 224], [3, 64, 288], [1, 1, 1]),
        ((4, 128, 1291), (1, 32, 1291), [0, 96, 0], [1, 128, 1291], [1, 1, 1]),  # broadcast
        # === 4D tensors ===
        # Medium volume, full slice
        ((2, 3, 128, 1291), (2, 3, 128, 1291), [0, 0, 0, 0], [2, 3, 128, 1291], [1, 1, 1, 1]),
        ((2, 3, 128, 1291), (1, 1, 128, 1291), [0, 0, 0, 0], [2, 3, 128, 1291], [1, 1, 1, 1]),  # broadcast
        # Partial slice in last 2 dims, tile-aligned
        ((2, 3, 128, 1291), (2, 3, 32, 64), [0, 0, 96, 96], [2, 3, 128, 160], [1, 1, 1, 1]),
        ((2, 3, 128, 1291), (1, 1, 32, 1291), [1, 2, 96, 0], [2, 3, 128, 1291], [1, 1, 1, 1]),  # broadcast
        # === 5D tensors ===
        # Medium volume, full slice
        ((2, 2, 3, 128, 1291), (2, 2, 3, 128, 1291), [0, 0, 0, 0, 0], [2, 2, 3, 128, 1291], [1, 1, 1, 1, 1]),
        (
            (2, 2, 3, 128, 1291),
            (1, 1, 1, 128, 1291),
            [0, 0, 0, 0, 0],
            [2, 2, 3, 128, 1291],
            [1, 1, 1, 1, 1],
        ),  # broadcast
        # Large volume, tile-aligned slice in last 2 dims
        ((2, 2, 3, 1024, 2048), (2, 2, 3, 64, 64), [0, 0, 0, 960, 1984], [2, 2, 3, 1024, 2048], [1, 1, 1, 1, 1]),
        (
            (2, 2, 3, 1024, 2048),
            (1, 1, 1, 64, 2048),
            [1, 1, 2, 960, 0],
            [2, 2, 3, 1024, 2048],
            [1, 1, 1, 1, 1],
        ),  # broadcast
        # === Lower-rank value broadcasting ===
        # 2D value into 5D last 2 dims
        ((1, 1, 3, 32, 64), (32, 64), [0, 0, 0, 0, 0], [1, 1, 3, 32, 64], [1, 1, 1, 1, 1]),
        # 3D input, 1D value → broadcast to last 2 dims (illegal unless input.shape[-2:] == value.shape)
        ((1, 1, 32), (32,), [0, 0, 0], [1, 1, 32], [1, 1, 1]),  # assumes final dim write only
        # 3D input, 2D value → broadcast into final 2 dims
        ((2, 32, 64), (32, 64), [0, 0, 0], [2, 32, 64], [1, 1, 1]),
        # ((2, 32, 64), (1, 64), [0, 0, 0], [2, 32, 64], [1, 1, 1]),  #FIXME: This case cannot be broadcast using tile
        # ((2, 32, 64), (32, 1), [0, 0, 0], [2, 32, 64], [1, 1, 1]),  #FIXME: This case cannot be broadcast using tile
        # 4D input, 2D value broadcast to final 2 dims
        ((3, 2, 32, 64), (32, 64), [0, 0, 0, 0], [3, 2, 32, 64], [1, 1, 1, 1]),
        # ((3, 2, 32, 64), (1, 64), [0, 0, 0, 0], [3, 2, 32, 64], [1, 1, 1, 1]),  #FIXME: This case cannot be broadcast using tile
        # ((3, 2, 32, 64), (32, 1), [0, 0, 0, 0], [3, 2, 32, 64], [1, 1, 1, 1]),    #FIXME: This case cannot be broadcast using tile
        # 5D input, 3D value broadcast to final 3 dims
        ((2, 1, 3, 32, 64), (3, 32, 64), [0, 0, 0, 0, 0], [2, 1, 3, 32, 64], [1, 1, 1, 1, 1]),
        ((2, 1, 3, 32, 64), (1, 32, 64), [0, 0, 0, 0, 0], [2, 1, 3, 32, 64], [1, 1, 1, 1, 1]),
    ],
)
def test_setitem_slice_only(
    device,
    inp_shape,
    inp_dtype,
    val_shape,
    val_dtype,
    begins,
    ends,
    steps,
    idx_dtype,
):
    setitem_slice(
        device, inp_shape, inp_dtype, val_shape, val_dtype, begins, ends, steps, idx_dtype, layout=ttnn.TILE_LAYOUT
    )


##############################################
#### Setitem with slice and index tensors ####
##############################################


def slice_and_index(inp_shape, val_shape, index_dims, indices, begins, ends, steps):
    """
    Helper to generate a test case for setitem with slices and index tensors.
    """
    for dim in range(len(inp_shape)):
        if dim in index_dims:
            begins[dim] = 0
            ends[dim] = inp_shape[dim]
            steps[dim] = 1
        else:
            if begins[dim] is None:
                begins[dim] = 0
            if ends[dim] is None:
                ends[dim] = inp_shape[dim]
            if steps[dim] is None:
                steps[dim] = 1

    return (inp_shape, val_shape, index_dims, indices, begins, ends, steps)


@pytest.mark.parametrize(
    "inp_shape, val_shape, index_dims, indices, begins, ends, steps",
    [
        # full slice on last two dims
        slice_and_index((2, 64, 64), (2, 64, 64), [0], [[0, 1]], [None, 0, 0], [None, 64, 64], [None, 1, 1]),
        # aligned 32x32 slice from offset
        slice_and_index((2, 96, 96), (2, 32, 32), [0], [[0, 1]], [None, 32, 32], [None, 64, 64], [None, 1, 1]),
        # full height slice, partial width
        slice_and_index((1, 128, 64), (1, 128, 32), [0], [[0]], [None, 0, 0], [None, 128, 32], [None, 1, 1]),
        # tile-aligned begin and size (not from zero)
        slice_and_index((2, 128, 128), (2, 64, 64), [0], [[0, 1]], [None, 32, 32], [None, 96, 96], [None, 1, 1]),
        # full slice on last dim only
        slice_and_index((1, 96, 256), (1, 64, 256), [0], [[0]], [None, 0, 0], [None, 64, 256], [None, 1, 1]),
        # large tensor, partial aligned slice
        slice_and_index(
            (4, 4096, 256), (4, 1024, 256), [0], [[0, 1, 2, 3]], [None, 0, 0], [None, 1024, 256], [None, 1, 1]
        ),
        # deep offset slice with tile alignment
        slice_and_index((2, 512, 512), (2, 64, 64), [0], [[0, 1]], [None, 448, 448], [None, 512, 512], [None, 1, 1]),
        # middle-aligned slice
        slice_and_index((2, 256, 256), (2, 64, 64), [0], [[0, 1]], [None, 64, 128], [None, 128, 192], [None, 1, 1]),
        # begin=0, slice_size aligned but not full
        slice_and_index((2, 64, 128), (2, 32, 64), [0], [[0, 1]], [None, 0, 0], [None, 32, 64], [None, 1, 1]),
        # : one slice full, one aligned
        slice_and_index((1, 128, 64), (1, 128, 32), [0], [[0]], [None, 0, 0], [None, 128, 32], [None, 1, 1]),
        # 64 x 30 -> 30 not divisible by 32 → full slice needed
        slice_and_index(
            (2, 64, 30),
            (2, 32, 30),
            [0],
            [[0, 1]],
            [None, 0, 0],
            [None, 32, 30],
            [None, 1, 1],
        ),
        # 50 x 50 → both not divisible by 32 → full slice required
        slice_and_index(
            (2, 50, 50),
            (2, 50, 50),
            [0],
            [[0, 1]],
            [None, 0, 0],
            [None, 50, 50],
            [None, 1, 1],
        ),
        # 2041 x 64 → 2041 not divisible by 32 → full slice for dim 1
        slice_and_index(
            (2, 2041, 64),
            (2, 2041, 32),
            [0],
            [[0, 1]],
            [None, 0, 0],
            [None, 2041, 32],
            [None, 1, 1],
        ),
        # 67 x 99 → both dimensions non-divisible
        slice_and_index(
            (1, 67, 99),
            (1, 67, 99),
            [0],
            [[0]],
            [None, 0, 0],
            [None, 67, 99],
            [None, 1, 1],
        ),
        # Only last dim non-divisible
        slice_and_index(
            (2, 64, 31),
            (2, 32, 31),
            [0],
            [[0, 1]],
            [None, 0, 0],
            [None, 32, 31],
            [None, 1, 1],
        ),
        # 33 x 256 (first misaligned)
        slice_and_index(
            (1, 33, 256),
            (1, 33, 128),
            [0],
            [[0]],
            [None, 0, 0],
            [None, 33, 128],
            [None, 1, 1],
        ),
        # Large odd shape
        slice_and_index(
            (2, 1025, 257),
            (2, 1025, 257),
            [0],
            [[0, 1]],
            [None, 0, 0],
            [None, 1025, 257],
            [None, 1, 1],
        ),
        # Narrow but unaligned
        slice_and_index(
            (1, 45, 17),
            (1, 45, 17),
            [0],
            [[0]],
            [None, 0, 0],
            [None, 45, 17],
            [None, 1, 1],
        ),
        # Edge case - both just under 32
        slice_and_index(
            (2, 31, 31),
            (2, 31, 31),
            [0],
            [[0, 1]],
            [None, 0, 0],
            [None, 31, 31],
            [None, 1, 1],
        ),
        # Mixed large + small non-divisible shapes
        slice_and_index(
            (2, 123, 4095),
            (2, 123, 4095),
            [0],
            [[0, 1]],
            [None, 0, 0],
            [None, 123, 4095],
            [None, 1, 1],
        ),
    ],
)
def test_setitem_with_index_tensors_and_slices(device, inp_shape, val_shape, index_dims, indices, begins, ends, steps):
    inp_dtype = i32
    val_dtype = i32
    idx_dtype = i32

    att, apt = make_input(inp_shape, dtype=inp_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    vtt, vpt = make_value(val_shape, dtype=val_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    # Convert indexing tensors
    idx_torch = [torch.tensor(idx, dtype=idx_dtype) for idx in indices]
    idx_ttnn = [ttnn.from_torch(t, device=device) for t in idx_torch]

    # Construct mixed indexing tuple for PyTorch
    full_index = []
    dim_map = {dim: t for dim, t in zip(index_dims, idx_torch)}
    for dim in range(len(inp_shape)):
        if dim in dim_map:
            full_index.append(dim_map[dim])
        else:
            b, e, s = begins[dim], ends[dim], steps[dim]
            full_index.append(slice(b, e, s))
    apt[tuple(full_index)] = vpt

    # Apply TTNN setitem
    ott = ttnn.bos_setitem(
        att,
        vtt,
        begins=begins,
        ends=ends,
        steps=steps,
        index_tensors=idx_ttnn,
        index_dims=index_dims,
    )

    result = ttnn.to_torch(ott)
    assert torch.allclose(result, apt), f"Setitem failed\nExpected:\n{apt}\nGot:\n{result}"
