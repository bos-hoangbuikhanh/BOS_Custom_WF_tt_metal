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
3. Validate setitem operations for various edge cases.
    - Invalid input count, index dtype, index length, layout.
    - Unbroadcastable value shape.
    - Invalid memory config.
    - Invalid slice parameters.
"""
import pytest
import torch
import ttnn

f16 = torch.float16
f32 = torch.float32
i32 = torch.int32
i64 = torch.int64
bf16 = torch.bfloat16
NUM_ITERS = 5

##################
#### Helper ######
##################


def make_input(shape=(4, 4), dtype=bf16, layout=ttnn.ROW_MAJOR_LAYOUT, device=None):
    tensor_pt = torch.zeros(shape, dtype=dtype)
    return ttnn.from_torch(tensor_pt, device=device, layout=layout), tensor_pt


def make_value(shape=(1, 4), dtype=bf16, layout=ttnn.ROW_MAJOR_LAYOUT, device=None):
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
    device,
    inp_shape,
    inp_dtype,
    val_shape,
    val_dtype,
    begins,
    ends,
    steps,
    idx_tensor,
    idx_dims,
    idx_dtype,
    layout=ttnn.ROW_MAJOR_LAYOUT,
):
    # prepare input & value
    att, apt = make_input(inp_shape, device=device, layout=layout, dtype=inp_dtype)
    vtt, vpt = make_value(val_shape, dtype=val_dtype, layout=layout, device=device)
    idx = torch.tensor(idx_tensor, dtype=idx_dtype) if idx_tensor is not None else None

    # --- Slice logic ---
    input_rank = len(inp_shape)
    slices = []
    adjusted_begins = list(begins)
    adjusted_ends = list(ends)
    adjusted_steps = list(steps)

    # build slices
    for b, e, s in zip(adjusted_begins, adjusted_ends, adjusted_steps):
        slices.append(slice(b, e, s))
    slices = tuple(slices)

    # call via the new slice-API
    ott = ttnn.bos_setitem(
        att,
        vtt,
        begins=begins,
        ends=ends,
        steps=steps,
        #    index_tensors=idx,
        #    index_dims=idx_dims,
    )
    result = ttnn.to_torch(ott)
    apt[slices] = vpt

    print(result, apt)

    # 5) compare
    assert torch.allclose(result, apt), "slice-API gave wrong result"

    import time

    for i in range(NUM_ITERS):
        att, apt = make_input(inp_shape, device=device, layout=layout, dtype=inp_dtype)
        vtt, vpt = make_value(val_shape, dtype=val_dtype, layout=layout, device=device)
        st = time.time()
        ott = ttnn.bos_setitem(
            att,
            vtt,
            begins=begins,
            ends=ends,
            steps=steps,
        )
        ttnn.synchronize_device(device)
        en = time.time()
        avg_exec_time = en - st
        print(f"Iter {i+1}/{NUM_ITERS} done, runtime: {avg_exec_time:.4f} s", end="\r")

    print(f"Average time: {avg_exec_time:.4f} s | {1/(avg_exec_time):.2f} FPS")


# Slice outer, full inner
@pytest.mark.parametrize("inp_dtype, val_dtype, idx_dtype", [(bf16, bf16, i32)])
@pytest.mark.parametrize(
    "inp_shape, val_shape, begins, ends, steps, idx_tensor, idx_dims",
    [
        # 4×4 → set rows 0–1
        ((4, 4), (2, 4), [0, 0], [2, 4], [1, 1], None, []),
        # 4×4 → every other row (0,2)
        ((4, 4), (2, 4), [0, 0], [4, 4], [2, 1], None, []),
        # 4×4 → full slice (all 4 rows)
        ((4, 4), (4, 4), [0, 0], [4, 4], [1, 1], None, []),
        # 4×4 → empty slice (row 2 to 2 → no rows)
        ((4, 4), (0, 4), [2, 0], [2, 4], [1, 1], None, []),
        # 4×4 → negative begin wraps: [-2..4) → rows [2,3]
        ((4, 4), (2, 4), [-2, 0], [4, 4], [1, 1], None, []),
        ((4, 4, 3), (2, 4, 3), [-2, 0, 0], [4, 4, 3], [1, 1, 1], None, []),
        # 5×4 → slice middle rows (1–4)
        ((5, 4), (3, 4), [1, 0], [4, 4], [1, 1], None, []),
        # 6×4 → every 2nd row (0,2,4)
        ((6, 4), (3, 4), [0, 0], [6, 4], [2, 1], None, []),
        # 3×4×5 → slice 1st dim [1:3], full 2nd, full 3rd
        ((3, 4, 5), (2, 4, 5), [1, 0, 0], [3, 4, 5], [1, 1, 1], None, []),
        # 3×4×5 → full slice (input shape = value shape)
        ((3, 4, 5), (3, 4, 5), [0, 0, 0], [3, 4, 5], [1, 1, 1], None, []),
        # 3×4×5 → full slice (empty value)
        ((3, 4, 5), (0, 4, 5), [0, 0, 0], [0, 4, 5], [1, 1, 1], None, []),
        # 3×4×5 → skip every 2 in outer dims
        ((3, 4, 5), (2, 4, 5), [0, 0, 0], [3, 4, 5], [2, 1, 1], None, []),
        # 3×4×5 → negative begin wraps: [-1, 0] → rows 2, full 2nd dim
        ((3, 4, 5), (1, 4, 5), [-1, 0, 0], [3, 4, 5], [1, 1, 1], None, []),
        # Large input shape, partial outer, full inner
        ((2000, 100), (1000, 100), [0, 0], [1000, 100], [1, 1], None, []),
        # Large input shape, partial outer, full inner
        ((1, 6, 10_000, 256), (1, 1, 10_000, 256), [0, 0, 0, 0], [1, 1, 10_000, 256], [1, 1, 1, 1], None, []),
        # TODO:
        # Reverse slices (negative step) not yet supported in kernel,
        # 5×4 → reverse slice on outer (rows 3→0, step -1)
        # ((5, 4), (3, 4), [3, 0], [0, 4], [-1, 1], None, []),
        # 3×4×5 → reverse 1st dim [2→0], full 2nd and 3rd
        # ((3, 4, 5), (2, 4, 5), [2, 0, 0], [0, 4, 5], [-1, 1, 1], None, []),
    ],
)
def test_setitem_slice_full_inner(
    device,
    inp_shape,
    inp_dtype,
    val_shape,
    val_dtype,
    begins,
    ends,
    steps,
    idx_tensor,
    idx_dims,
    idx_dtype,
):
    setitem_slice(
        device,
        inp_shape,
        inp_dtype,
        val_shape,
        val_dtype,
        begins,
        ends,
        steps,
        idx_tensor,
        idx_dims,
        idx_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


# Slice outer, Slice inner
@pytest.mark.parametrize("inp_dtype, val_dtype, idx_dtype", [(bf16, bf16, i32)])
@pytest.mark.parametrize(
    "inp_shape, val_shape, begins, ends, steps, idx_tensor, idx_dims",
    [
        # 4×4 → assign rows 0–1, cols 1–3 (2x2)
        ((4, 4), (2, 2), [0, 1], [2, 3], [1, 1], None, []),
        # 4×4 → assign rows 2–4, cols 0–2 (2x2)
        ((4, 4), (2, 2), [2, 0], [4, 2], [1, 1], None, []),
        # 4×4 → every other row, cols 1–4: shape (2,3)
        ((4, 4), (2, 3), [0, 1], [4, 4], [2, 1], None, []),
        # 5×4 → assign rows 1–4, cols 0–2 (3x2)
        ((5, 4), (3, 2), [1, 0], [4, 2], [1, 1], None, []),
        # 6×4 → rows [0,2,4], cols 1–4 (3x3)
        ((6, 4), (3, 3), [0, 1], [6, 4], [2, 1], None, []),
        # 3×4×5 → first dim [1:3], full 2nd dim, partial 3rd dim
        ((3, 4, 5), (2, 4, 2), [1, 0, 0], [3, 4, 2], [1, 1, 1], None, []),
        # 3×4×5 → full outer, partial last dim
        ((3, 4, 5), (3, 4, 2), [0, 0, 1], [3, 4, 3], [1, 1, 1], None, []),
        # 2×4×6 → slice middle of last dim [2:5] (length 3)
        ((2, 4, 6), (2, 4, 3), [0, 0, 2], [2, 4, 5], [1, 1, 1], None, []),
        # # 2×4×6 → partial inner-most (3 of 6)
        ((2, 4, 6), (2, 4, 3), [0, 0, 0], [2, 4, 6], [1, 1, 2], None, []),
        # 3×4×5 → partial last dim: step=2, cols [0,2,4] - inner slice with strided index
        ((3, 4, 5), (2, 4, 3), [1, 0, 0], [3, 4, 5], [1, 1, 2], None, []),
        # Large input shape, partial outer, partial inner
        ((2000, 500), (1000, 300), [0, 0], [1000, 300], [1, 1], None, []),
    ],
)
def test_setitem_slice_misalignment_inner_dim(
    device, inp_shape, inp_dtype, val_shape, val_dtype, begins, ends, steps, idx_tensor, idx_dims, idx_dtype
):
    setitem_slice(
        device,
        inp_shape,
        inp_dtype,
        val_shape,
        val_dtype,
        begins,
        ends,
        steps,
        idx_tensor,
        idx_dims,
        idx_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


# Slice outer, Slice inner
@pytest.mark.parametrize("inp_dtype, val_dtype, idx_dtype", [(bf16, bf16, i32)])
@pytest.mark.parametrize(
    "inp_shape, val_shape, begins, ends, steps, idx_tensor, idx_dims",
    [
        # broadcasted: 2×4×6 → partial inner-most (3 of 6)
        ((2, 4, 6), (2, 4, 1), [0, 0, 0], [2, 4, 6], [1, 1, 1], None, []),
        # broadcasted: inner-most dim (6 → 1), applied across [2,4]
        # ((2, 4, 6), (2, 4, 1), [0, 0, 0], [2, 4, 6], [1, 1, 2], None, []), #todo
        # broadcasted: middle dim (4 → 1)
        ((2, 4, 6), (2, 1, 6), [0, 0, 0], [2, 4, 6], [1, 1, 1], None, []),
        # broadcasted: outer-most dim (2 → 1)
        ((2, 4, 6), (1, 4, 6), [0, 0, 0], [2, 4, 6], [1, 1, 1], None, []),
        # broadcasted: 2×4×6 input, scalar value (1×1×1)
        ((2, 4, 6), (1, 1, 1), [0, 0, 0], [2, 4, 6], [1, 1, 1], None, []),
        # broadcasted: 3×1×6 (broadcast along middle dim)
        ((3, 4, 6), (3, 1, 6), [0, 0, 0], [3, 4, 6], [1, 1, 1], None, []),
        # broadcasted: 1×4×1 → fills outer and inner dims
        # ((2, 4, 6), (1, 4, 1), [0, 0, 0], [2, 4, 6], [1, 1, 2], None, []), #todo
        # broadcasted: 1×1×6 → broadcast both outer and middle dims
        ((2, 4, 6), (1, 1, 6), [0, 0, 0], [2, 4, 6], [1, 1, 1], None, []),
        # broadcasted: 2×1×1 → keep outer dim, broadcast inner two
        # ((2, 4, 6), (2, 1, 1), [0, 0, 0], [2, 4, 6], [1, 1, 3], None, []), #todo
        # broadcasted: 1×4×3 → broadcast outer, partial inner slice
        ((2, 4, 6), (1, 4, 3), [0, 0, 1], [2, 4, 6], [1, 1, 2], None, []),
    ],
)
def test_setitem_slice_broadcasting(
    device, inp_shape, inp_dtype, val_shape, val_dtype, begins, ends, steps, idx_tensor, idx_dims, idx_dtype
):
    setitem_slice(
        device,
        inp_shape,
        inp_dtype,
        val_shape,
        val_dtype,
        begins,
        ends,
        steps,
        idx_tensor,
        idx_dims,
        idx_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


# Broadcasting where value has lower rank than input
@pytest.mark.parametrize("inp_dtype, val_dtype, idx_dtype", [(bf16, bf16, i32)])
@pytest.mark.parametrize(
    "inp_shape, val_shape, begins, ends, steps, idx_tensor, idx_dims",
    [
        # val is scalar (rank-0) → broadcast to full slice region
        ((2, 4, 6), (), [0, 0, 0], [2, 4, 6], [1, 1, 1], None, []),
        # val is 1D → broadcast to match last dim
        ((2, 4, 6), (6,), [0, 0, 0], [2, 4, 6], [1, 1, 1], None, []),
        # val is 2D, matches innermost dims: 4×6 → broadcast outer
        ((2, 4, 6), (4, 6), [0, 0, 0], [2, 4, 6], [1, 1, 1], None, []),
        # val is 1D, partial inner dim slice (step size > 1)
        ((2, 4, 6), (3,), [0, 0, 1], [2, 4, 6], [1, 1, 2], None, []),
        # val is scalar, partial slice across all dims
        ((2, 4, 6), (), [0, 1, 2], [2, 4, 6], [1, 2, 2], None, []),
    ],
)
def test_setitem_slice_broadcasting_lower_rank(
    device, inp_shape, inp_dtype, val_shape, val_dtype, begins, ends, steps, idx_tensor, idx_dims, idx_dtype
):
    setitem_slice(
        device,
        inp_shape,
        inp_dtype,
        val_shape,
        val_dtype,
        begins,
        ends,
        steps,
        idx_tensor,
        idx_dims,
        idx_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


@pytest.mark.parametrize(
    "inp_shape, val_shape, index_dims, indices",
    [
        # Basic index tensor cases
        ((4, 4), (2, 4), [0], [[0, 2]]),  # rows 0 and 2
        ((4, 4), (2, 4), [0], [[3, 1]]),  # reversed order
        # 3D tensor with 2 index tensors
        ((3, 4, 5), (2, 5), [0, 1], [[0, 2], [1, 3]]),  # select elements [0,1] and [2,3]
        # Broadcasting: 1D value
        # ((4, 4), (4,), [0], [[0, 2]]),                    # (4,) broadcast to (2, 4)
        ((4, 4), (1,), [0], [[1, 3]]),  # scalar broadcast to (2, 4)
        # 3D broadcasting inner dim
        ((4, 3, 5), (1, 5), [0, 1], [[1, 2], [-1, -2]]),  # (1,5) to (2,5)
        # Full scalar broadcast
        ((4, 5), (), [0], [[0, 2]]),  # scalar to (2,5)
        # Lower-rank broadcasting
        ((2, 4, 6), (6,), [0], [[0, 1]]),  # 1D value → broadcast to 2x4x6
        # 3D tensor, full scalar
        ((2, 4, 6), (), [0], [[0, 1]]),
        # Indexing all dims
        ((2, 2), (1,), [0, 1], [[1], [1]]),  # single element overwrite
        # Empty index (should be a no-op)
        # ((4, 4), (0, 4), [0], [[]]),           #FIXME: hanging
    ],
)
def test_setitem_with_index_tensors_only(device, inp_shape, val_shape, index_dims, indices):
    inp_dtype = bf16
    val_dtype = bf16
    idx_dtype = i32

    att, apt = make_input(inp_shape, dtype=inp_dtype, device=device)
    vtt, vpt = make_value(val_shape, dtype=val_dtype, device=device)

    idx_torch = [torch.tensor(idx, dtype=idx_dtype) for idx in indices]
    idx_ttnn = [ttnn.from_torch(t, device=device) for t in idx_torch]

    # Dynamically construct the index tuple
    index_tuple = tuple(indices)
    apt[index_tuple] = vpt

    # ttnn.bos_setitem API call
    ott = ttnn.bos_setitem(
        att,
        vtt,
        begins=[0] * len(inp_shape),
        ends=list(inp_shape),
        steps=[1] * len(inp_shape),
        index_tensors=idx_ttnn,
        index_dims=index_dims,
    )

    result = ttnn.to_torch(ott)
    assert torch.allclose(result, apt), f"Setitem with index tensors failed\nExpected:\n{apt}\nGot:\n{result}"


def get_slice_with_index_test_case(inp_shape, val_shape, index_dims, indices, begins, ends, steps):
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
        # # Index first dim with tensor, slice second dim
        # get_slice_with_index_test_case((4, 6), (2, 3), [0], [[1, 3]], [None, 2], [None, 5], [None, 1]),  # input[ [1,3], 2:5 ] = val
        # # Index second dim with tensor, slice first dim
        # get_slice_with_index_test_case((5, 4), (2, 2), [1], [[0, 2]], [1, None], [3, None], [1, None]),  # input[1:3, [0,2]] = val
        # # 3D case: tensor index on first dim, slice on second and third
        # get_slice_with_index_test_case((4, 5, 6), (2, 2, 3), [0], [[0, 2]], [None, 1, 2], [None, 3, 5], [None, 1, 1]),  # input[ [0,2], 1:3, 2:5 ] = val
        # # 3D case: tensor index on second dim, slice on others
        # get_slice_with_index_test_case((3, 6, 4), (2, 2, 4), [1], [[1, 3]], [0, None, 0], [2, None, 4], [1, None, 1]),  # input[0:2, [1,3], :] = val
        # # # All slices except last dim is indexed
        # get_slice_with_index_test_case((4, 5, 6), (2, 2, 1), [2], [[0, 2]], [1, 2, None], [3, 4, None], [1, 1, None]),  # input[1:3, 2:4, [0,2]] = val
        # get_slice_with_index_test_case((1, 30, 256), (1, 30, 256), [1], [[i for i in range(0, 30, 1)]], [0, 0, 0], [None, None, None], [1, 1, 1]),
        # # Large tensor and large indexing tensor
        # get_slice_with_index_test_case((1, 10000, 256), (1, 3000, 256), [1], [[i for i in range(0, 3000, 1)]], [0, 0, 0], [None, None, None], [1, 1, 1]),
        get_slice_with_index_test_case(
            (1, 10_000, 4, 2),
            (1, 10_000, 4, 2),
            [1],
            [[i for i in range(0, 10_000, 1)]],
            [0, 0, 0, 0],
            [None, None, None, None],
            [1, 1, 1, 1],
        ),
    ],
)
def test_setitem_with_index_tensors_and_slices(device, inp_shape, val_shape, index_dims, indices, begins, ends, steps):
    inp_dtype = i32
    val_dtype = i32
    idx_dtype = i32

    att, apt = make_input(inp_shape, dtype=inp_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    vtt, vpt = make_value(val_shape, dtype=val_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

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

    import time

    for i in range(NUM_ITERS):
        att, apt = make_input(inp_shape, device=device, dtype=inp_dtype)
        vtt, vpt = make_value(val_shape, dtype=val_dtype, device=device)
        st = time.time()
        ott = ttnn.bos_setitem(
            att,
            vtt,
            begins=begins,
            ends=ends,
            steps=steps,
            index_tensors=idx_ttnn,
            index_dims=index_dims,
        )
        ttnn.synchronize_device(device)
        en = time.time()
        avg_exec_time = en - st
        print(f"Iter {i+1}/{NUM_ITERS} done, runtime: {avg_exec_time:.4f} s", end="\r")

    print(f"Average time: {avg_exec_time:.4f} s | {1/(avg_exec_time):.2f} FPS")


##########################
#### Validation Tests ####
##########################


# Validation: wrong number of inputs still errors
def test_setitem_invalid_input_count(device):
    att, _ = make_input(device=device)
    vtt, _ = make_value(device=device)
    # too few args (must have exactly 4)
    with raises_with_msg(TypeError):
        ttnn.bos_setitem(att, vtt)


# Validation: wrong dtype for outer index
@pytest.mark.parametrize(
    "inp_dtype, value_dtype",
    [
        (i64, f32),
        (i32, f32),
        (i64, bf16),
    ],
)
def test_setitem_invalid_value_dtype(device, inp_dtype, value_dtype):
    att, _ = make_input(device=device, dtype=inp_dtype)
    vtt, _ = make_value(device=device, dtype=value_dtype)
    # outer idx has invalid dtype
    idx = torch.tensor([0], dtype=i32)
    idt = idx.tolist()
    # inner idx (valid)
    inner_idx = torch.arange(4, dtype=i32)
    inner_t = inner_idx.tolist()

    with raises_with_msg(RuntimeError):
        ttnn.bos_setitem(att, vtt, idt, inner_t)


# Validation: unbroadcastable value shape
@pytest.mark.parametrize(
    "inp_shape, value_shape",
    [
        ([3, 5], [2, 5]),
        ([4, 5], [2, 5]),
        ([4, 5], [5, 5]),
    ],
)
def test_setitem_unbroadcastable_check(device, inp_shape, value_shape):
    att, _ = make_input(shape=inp_shape, device=device)
    vtt, _ = make_value(shape=value_shape, device=device)
    it = ttnn.from_torch(torch.tensor([0], dtype=i32), device=device)
    begin = [0] * len(inp_shape)
    end = [i for i in inp_shape]

    with raises_with_msg(RuntimeError, ERROR_MAPPING["NOT_BROADCASTABLE"]):
        ttnn.bos_setitem(att, vtt, begins=begin, ends=end)


# Validation: outer index length exceeds dimension count
def test_setitem_index_length_exceeds(device):
    att, _ = make_input(device=device)
    vtt = ttnn.from_torch(torch.ones((5, 4), dtype=bf16), device=device)
    idx = torch.arange(5, dtype=i32)
    it = idx.tolist()
    inner_idx = torch.arange(4, dtype=i32)
    inner_t = inner_idx.tolist()

    with raises_with_msg(RuntimeError):
        ttnn.bos_setitem(att, vtt, it, inner_t)


##########################
## Non-pytest Functions ##
##########################


def python_test_setitem_idx(device, inp_shape=[4, 4], val_shape=[1, 4], outer_idx=[0], inner_idx=[0], **kwargs):
    att, apt = make_input(inp_shape, device=device, dtype=bf16, **kwargs)
    # att = ttnn.to_memory_config(att, memory_config=ttnn.L1_MEMORY_CONFIG) # if glocal cb for input
    vtt, vpt = make_value(val_shape, dtype=bf16, device=device, **kwargs)
    outer = torch.tensor(outer_idx, dtype=i32)
    inner = torch.tensor(inner_idx, dtype=i32)
    out_t = ttnn.from_torch(outer, device=device)
    in_t = ttnn.from_torch(inner, device=device)
    print(att)
    print(vtt)
    ttnn.bos_setitem(att, vtt, out_t, in_t)
    print(att)


def python_test_setitem_slices(device, inp_shape=[4, 4], val_shape=[1, 4], begins=[0], ends=[0], steps=None, **kwargs):
    att, apt = make_input(inp_shape, device=device, dtype=bf16, **kwargs)
    # att = ttnn.to_memory_config(att, memory_config=ttnn.L1_MEMORY_CONFIG) # if glocal cb for input
    vtt, vpt = make_value(val_shape, dtype=bf16, device=device, **kwargs)
    print(att)
    print(vtt)
    ttnn.bos_setitem(att, vtt, begins, ends, steps)
    print(att)


# device = ttnn.open_device(device_id=0)

# inp, inp_pt = make_input(shape=(2, 4), device=device)
# val, val_pt = make_value(shape=(1, 4), device=device)
# idx, idx_pt = make_index([0], device=device)

# inp[idx] = val

# test_setitem_with_index_tensors_and_slices(
#     device,
#     *get_slice_with_index_test_case((4, 5, 6), (2, 2, 1), [2], [[0, 2]], [1, 2, None], [3, 4, None], [1, 1, None])
# )
# python_test_setitem_slices(device, inp_shape=[1024, 1024], val_shape=[512, 512], begins=[256, 256], ends=[768, 768], steps=[1, 1], layout=ttnn.TILE_LAYOUT)
